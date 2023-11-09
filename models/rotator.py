from typing import Optional
import numpy as np
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from models.single_cls import SingleClassifier


class Rotator(SingleClassifier):
    state_fc: nn.Module

    def __init__(self, cfg, train_ds, val_ds, freeze_mapping_layer=True):
        self.estimate_init_state = False
        self.estimate_final_state = False

        super().__init__(cfg, train_ds, val_ds, freeze_mapping_layer)

    def build_model(self):
        self.estimate_init_state = self.cfg["train"]["rotator"]["estimate_init_state"]
        self.estimate_final_state = self.cfg["train"]["rotator"]["estimate_final_state"]

        # state estimation layers
        self.state_fc = nn.Sequential(
            nn.Linear(self.img_feat_dim, 512),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Linear(64, 8),
        )

        super().build_model()

    def forward(self, batch, batch_idx, view_override=None):
        (model1_n_feats, model2_n_feats), lang_feats, ans, _, _, is_visual = batch
        bs = model1_n_feats.shape[0]

        # to device
        model1_n_feats = model1_n_feats.to(device=self.device).float()
        model2_n_feats = model2_n_feats.to(device=self.device).float()
        lang_feats = lang_feats.to(device=self.device).float()

        # normalize
        if self.cfg["train"]["normalize_feats"]:
            model1_n_feats = model1_n_feats / model1_n_feats.norm(dim=-1, keepdim=True)
            model2_n_feats = model2_n_feats / model2_n_feats.norm(dim=-1, keepdim=True)
            lang_feats = lang_feats / lang_feats.norm(dim=-1, keepdim=True)

        # mapping layers
        lang_feats = self.lang_fc(lang_feats)
        model1_n_feats = self.img_fc(model1_n_feats)
        model2_n_feats = self.img_fc(model2_n_feats)

        # estimate current view
        init_view_estimation = self.estimate_view(
            model1_n_feats,
            model2_n_feats,
            lang_feats,            
            view_override,
        )

        # output variables from state estimation
        init_views1 = init_view_estimation["views1"]
        init_views2 = init_view_estimation["views2"]

        est_init_views1 = init_view_estimation["est_views1"]
        est_init_views2 = init_view_estimation["est_views2"]
        
        loss = init_view_estimation["loss"] if self.estimate_init_state else torch.tensor(0)

        output_dict = {}

        if view_override is None:
            # choose features of ramdomly sampling viewpoints
            (
                model1_feats,
                model2_feats,
                rotated_views1,
                rotated_views2,
            ) = self.choose_feats_from_random_views(
                bs, model1_n_feats, model2_n_feats, init_views1, init_views2
            )

            # estimate second view before performing prediction
            final_state_estimation = self.estimate_view(
                model1_n_feats,
                model2_n_feats,
                lang_feats,                
                torch.stack([rotated_views1, rotated_views2]),
            )
            est_final_views1 = final_state_estimation["est_views1"]
            est_final_views2 = final_state_estimation["est_views2"]

            # put rotated views on device
            rotated_views1 = rotated_views1.to(device=self.device).int()
            rotated_views2 = rotated_views2.to(device=self.device).int()
            est_final_view1_corrects = int(
                torch.count_nonzero(est_final_views1 == rotated_views1)
            )
            est_final_view2_corrects = int(
                torch.count_nonzero(est_final_views2 == rotated_views2)
            )
            total_correct_final_view_est = (
                est_final_view1_corrects + est_final_view2_corrects
            )
            output_dict = output_dict | {
                "total_correct_final_view_est": total_correct_final_view_est,
                "est_final_views1": est_final_views1,
                "est_final_views2": est_final_views2,
            }
            # loss += final_state_estimation['loss']
        else:
            model1_feats = model1_n_feats[:, view_override, :]
            model2_feats = model2_n_feats[:, view_override, :]

        # classifier probablities chosen features
        img1_chosen_prob = self.cls_fc(torch.cat([model1_feats, lang_feats], dim=-1))
        img2_chosen_prob = self.cls_fc(torch.cat([model2_feats, lang_feats], dim=-1))

        # classifier loss
        raw_probs = torch.cat([img1_chosen_prob, img2_chosen_prob], dim=-1)
        probs = F.softmax(raw_probs, dim=-1)
        num_steps = torch.ones((bs)).to(dtype=torch.long, device=lang_feats.device) * 2

        test_mode = ans[0] == -1
        if not test_mode:
            # classifier loss
            cls_labels = F.one_hot(ans, num_classes=2)
            cls_loss_weight = self.cfg["train"]["loss"]["cls_weight"]
            loss += (
                self.smoothed_cross_entropy(raw_probs, cls_labels)
            ) * cls_loss_weight

            # state estimation accuracy
            est_init_view1_corrects = int(
                torch.count_nonzero(est_init_views1 == init_views1)
            )
            est_init_view2_corrects = int(
                torch.count_nonzero(est_init_views2 == init_views2)
            )
            total_correct_init_view_est = (
                est_init_view1_corrects + est_init_view2_corrects
            )

            # state estimation errors
            est_err = torch.cat(
                [
                    self.modulo_views(init_views1 - est_init_views1).abs().float(),
                    self.modulo_views(init_views2 - est_init_views2).abs().float(),
                ]
            )

            if view_override is None:
                est_err += torch.cat(
                    [
                        self.modulo_views(rotated_views1 - est_final_views1)
                        .abs()
                        .float(),
                        self.modulo_views(rotated_views2 - est_final_views2)
                        .abs()
                        .float(),
                    ]
                )
            est_err = est_err.mean()

            return output_dict | {
                "probs": probs,
                "action_loss": loss,
                "labels": cls_labels,
                "is_visual": is_visual,
                "num_steps": num_steps,
                "total_correct_init_view_est": total_correct_init_view_est,
                "est_error": est_err,
                "est_init_views1": est_init_views1,
                "est_init_views2": est_init_views2,
            }
        else:
            return {
                "probs": probs,
                "num_steps": num_steps,
            }

    def estimate_view(
        self,
        model1_feats: torch.Tensor,
        model2_feats: torch.Tensor,
        lang_feats: torch.Tensor,        
        view_override: Optional[int | torch.Tensor] = None,
    ):
        bs = model1_feats.shape[0]        

        views_model1 = (
            torch.randint(self.num_views, (bs,))
            if view_override is None
            else torch.full((bs,), view_override)
            if isinstance(view_override, int)
            else view_override[0]
        ).cuda()
        views_model2 = (
            torch.randint(self.num_views, (bs,))
            if view_override is None
            else torch.full((bs,), view_override)
            if isinstance(view_override, int)
            else view_override[1]
        ).cuda()

        # init features
        model1_rnd_feats = torch.stack(
            [model1_feats[i, views_model1[i], :] for i in range(bs)]
        )
        model2_rnd_feats = torch.stack(
            [model2_feats[i, views_model2[i], :] for i in range(bs)]
        )

        gt_views1 = F.one_hot(
            views_model1.to(torch.int64), num_classes=self.num_views
        )
        gt_views2 = F.one_hot(
            views_model2.to(torch.int64), num_classes=self.num_views
        )

        # state estimator
        est_views_logits1 = self.state_fc(model1_rnd_feats)
        est_views_logits2 = self.state_fc(model2_rnd_feats)

        outputs = {
            "img1_n_feats": model1_feats,
            "img2_n_feats": model2_feats,
            "lang_feats": lang_feats,            
            "loss": 0,
            "views1": views_model1,
            "views2": views_model2,
            "est_views1": F.softmax(est_views_logits1, dim=-1).argmax(-1),
            "est_views2": F.softmax(est_views_logits2, dim=-1).argmax(-1),
        }

        if self.training:
            # state estimation loss
            est_loss_weight = self.cfg["train"]["loss"]["est_weight"]
            outputs["loss"] = (
                (
                    self.smoothed_cross_entropy(est_views_logits1, gt_views1)
                    + self.smoothed_cross_entropy(
                        est_views_logits2, gt_views2
                    )
                )
                / 2
            ) * est_loss_weight

        return outputs

    def modulo_views(self, views):
        bs = views.shape[0]
        modulo_views = torch.zeros_like(views)
        for b in range(bs):
            view = views[b]

            if view < 4 and view >= -4:
                modulo_views[b] = view
            elif view >= 4:
                modulo_views[b] = -4 + (view % 4)
            elif view < -4:
                modulo_views[b] = 4 - (abs(view) % 4)
        return modulo_views

    def choose_feats_from_random_views(
        self,
        bs,
        model1_n_feats,
        model2_n_feats,
        init_views1,
        init_views2,
        view_override: Optional[int] = None,
    ):
        if view_override is not None:
            return (
                model1_n_feats[:, view_override, :],
                model2_n_feats[:, view_override, :],
                torch.tensor([view_override]),
                torch.tensor([view_override]),
            )

        rand_next_views = torch.randint(self.num_views, (2, bs))
        img1_chosen_feats = torch.stack(
            [
                model1_n_feats[i, [init_views1[i], rand_next_views[0, i]], :].max(
                    dim=-2
                )[0]
                for i in range(bs)
            ]
        )
        img2_chosen_feats = torch.stack(
            [
                model2_n_feats[i, [init_views2[i], rand_next_views[1, i]], :].max(
                    dim=-2
                )[0]
                for i in range(bs)
            ]
        )
        return (
            img1_chosen_feats,
            img2_chosen_feats,
            rand_next_views[0],
            rand_next_views[1],
        )

    def compute_metrics(
        self,
        labels,
        loss,
        probs,
        visual,
        num_steps,
        total_correct_init_view_est,
        total_correct_final_view_est,
    ):
        batch_size = probs.shape[0]
        val_total, val_correct, val_pl_correct = 0, 0, 0.0
        visual_total, visual_correct, pl_visual_correct = 0, 0, 0.0
        nonvis_total, nonvis_correct, pl_nonvis_correct = 0, 0, 0.0
        for b in range(batch_size):
            correct = self.check_correct(b, labels, probs)

            if correct:
                val_correct += 1
                val_pl_correct += 1.0 / num_steps[b]
            val_total += 1

            if bool(visual[b]):
                if correct:
                    visual_correct += 1
                    pl_visual_correct += 1.0 / num_steps[b]
                visual_total += 1
            else:
                if correct:
                    nonvis_correct += 1
                    pl_nonvis_correct += 1.0 / num_steps[b]
                nonvis_total += 1

        correct_ests = total_correct_init_view_est + total_correct_final_view_est
        total_rots = 2 * batch_size

        val_acc = float(val_correct) / val_total
        val_pl_acc = float(val_pl_correct) / val_total
        val_visual_acc = float(visual_correct) / visual_total
        val_pl_visual_acc = float(pl_visual_correct) / visual_total
        val_nonvis_acc = float(nonvis_correct) / nonvis_total
        val_pl_nonvis_acc = float(pl_nonvis_correct) / nonvis_total
        val_est_init_err = (
            total_rots - float(total_correct_init_view_est)
        ) / total_rots
        val_est_final_err = (
            total_rots - float(total_correct_final_view_est)
        ) / total_rots
        val_est_err = (2 * total_rots - float(correct_ests)) / (2 * total_rots)

        return dict(
            val_loss=loss,
            val_acc=val_acc,
            val_pl_acc=val_pl_acc,
            val_correct=val_correct,
            val_pl_correct=val_pl_correct,
            val_total=val_total,
            val_visual_acc=val_visual_acc,
            val_pl_visual_acc=val_pl_visual_acc,
            val_visual_correct=visual_correct,
            val_pl_visual_correct=pl_visual_correct,
            val_visual_total=visual_total,
            val_nonvis_acc=val_nonvis_acc,
            val_pl_nonvis_acc=val_pl_nonvis_acc,
            val_nonvis_correct=nonvis_correct,
            val_pl_nonvis_correct=pl_nonvis_correct,
            val_nonvis_total=nonvis_total,
            val_est_init_err=val_est_init_err,
            val_est_final_err=val_est_final_err,
            val_est_err=val_est_err,
        )

    def training_step(self, batch, batch_idx):
        out = self.forward(batch, batch_idx)

        if self.log_data:
            wandb.log(
                {
                    "tr/loss": out["action_loss"],
                }
            )

        # self.log_dict(
        #     {
        #         "train/action_loss": out["action_loss"],
        #         "train/total_correct_init_view_est": out["total_correct_init_view_est"],
        #         "train/total_correct_final_view_est": out[
        #             "total_correct_final_view_est"
        #         ],
        #         # "train/total_correct_best_view_est": out["total_correct_best_view_est"],
        #         "train/est_error": out["est_error"],
        #         # "train/caption_view_prediction_error": out[
        #         #     "caption_view_prediction_error"
        #         # ],
        #         "train/init_est_view_loss": out["init_est_view_loss"],
        #         # "train/init_est_caption_view_loss": out["init_est_caption_view_loss"],
        #         "train/final_est_view_loss": out["final_est_view_loss"],
        #         # "train/final_est_caption_view_loss": out["final_est_caption_view_loss"],
        #     }
        # )

        return dict(loss=out["action_loss"])

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self.val_all_outputs = []

    # TODO: modify this to really use the two random views instead of all views..
    # imagine: without adaption, all views are required to represent the object. with a specification
    # towards the vantage point (as the rotator and predictor aim to implement), the representation
    # of the objects from all other views might suffer.
    def validation_step(self, batch, batch_idx):
        all_view_results = {}
        for view in range(self.num_views):
            out = self.forward(batch, batch_idx, view_override=view)
            losses = self._criterion(out)

            loss = losses["loss"]
            probs = out["probs"]
            labels = out["labels"]
            visual = out["is_visual"]
            num_steps = out["num_steps"]
            # total_correct_init_view_est = out["total_correct_init_view_est"]

            metrics = self.compute_metrics(
                labels,
                loss,
                probs,
                visual,
                num_steps,
                0,
                0,
            )

            all_view_results[view] = metrics

        # TODO: somehow collect the distribution of the caption view prediction
        out = self.forward(batch, batch_idx)

        # losses
        losses = self._criterion(out)

        loss = losses["loss"]
        probs = out["probs"]
        labels = out["labels"]
        visual = out["is_visual"]
        num_steps = out["num_steps"]
        # total_correct_init_view_est = out["total_correct_init_view_est"]
        # total_correct_final_view_est = out["total_correct_final_view_est"]

        metrics = self.compute_metrics(
            labels,
            loss,
            probs,
            visual,
            num_steps,
            0,
            0,
        )

        results = dict(
            all_view_results=all_view_results,
            val_loss=metrics["val_loss"],
            val_acc=metrics["val_acc"],
            val_correct=metrics["val_correct"],
            val_total=metrics["val_total"],
            val_visual_correct=metrics["val_visual_correct"],
            val_visual_total=metrics["val_visual_total"],
            val_nonvis_correct=metrics["val_nonvis_correct"],
            val_nonvis_total=metrics["val_nonvis_total"],
        )
        self.val_all_outputs.append(results)
        return results

    def on_validation_epoch_end(self) -> None:
        # super().on_validation_epoch_end()

        mode = "vl"
        n_view_res = {}
        views = list(range(self.num_views))

        if len(self.val_all_outputs) < 1:
            return

        sanity_check = True
        for view in views:
            view_res = {
                "val_loss": 0.0,
                "val_correct": 0,
                "val_pl_correct": 0,
                "val_total": 0,
                "val_visual_correct": 0,
                "val_pl_visual_correct": 0,
                "val_visual_total": 0,
                "val_nonvis_correct": 0,
                "val_pl_nonvis_correct": 0,
                "val_nonvis_total": 0,
                "val_est_init_err": 0.0,
                "val_est_final_err": 0.0,
                "val_est_err": 0.0,
            }

            for output in self.val_all_outputs:
                metrics = output["all_view_results"][view]

                view_res["val_loss"] += metrics["val_loss"].item()

                view_res["val_correct"] += metrics["val_correct"]
                view_res["val_pl_correct"] += int(metrics["val_pl_correct"])
                view_res["val_total"] += metrics["val_total"]

                view_res["val_visual_correct"] += metrics["val_visual_correct"]
                view_res["val_pl_visual_correct"] += int(
                    metrics["val_pl_visual_correct"]
                )
                view_res["val_visual_total"] += metrics["val_visual_total"]

                view_res["val_nonvis_correct"] += metrics["val_nonvis_correct"]
                view_res["val_pl_nonvis_correct"] += int(
                    metrics["val_pl_nonvis_correct"]
                )
                view_res["val_nonvis_total"] += metrics["val_nonvis_total"]

                view_res["val_est_init_err"] += metrics["val_est_init_err"]
                view_res["val_est_final_err"] += metrics["val_est_final_err"]
                view_res["val_est_err"] += metrics["val_est_err"]            

            view_res["val_loss"] = float(view_res["val_loss"]) / len(
                self.val_all_outputs
            )
            
            view_res["val_acc"] = float(view_res["val_correct"]) / view_res["val_total"]
            view_res["val_pl_acc"] = (
                float(view_res["val_pl_correct"]) / view_res["val_total"]
            )
            if view_res["val_total"] > 128:
                sanity_check = False

            view_res["val_visual_acc"] = (
                float(view_res["val_visual_correct"]) / view_res["val_visual_total"]
            )
            view_res["val_pl_visual_acc"] = (
                float(view_res["val_pl_visual_correct"]) / view_res["val_visual_total"]
            )

            view_res["val_nonvis_acc"] = (
                float(view_res["val_nonvis_correct"]) / view_res["val_nonvis_total"]
            )
            view_res["val_pl_nonvis_acc"] = (
                float(view_res["val_pl_nonvis_correct"]) / view_res["val_nonvis_total"]
            )

            view_res["val_est_init_err"] = float(view_res["val_est_init_err"]) / len(
                self.val_all_outputs
            )
            view_res["val_est_final_err"] = float(view_res["val_est_final_err"]) / len(
                self.val_all_outputs
            )
            view_res["val_est_err"] = float(view_res["val_est_err"]) / len(
                self.val_all_outputs
            )

            n_view_res[view] = view_res

        mean_val_loss = np.mean([r["val_loss"] for r in n_view_res.values()])

        # WARNING: I changed this to consider the random/best views, not ALL view results
        val_acc = sum([o["val_correct"] for o in self.val_all_outputs]) / float(sum([o["val_total"] for o in self.val_all_outputs]))
        val_visual_acc = sum([o["val_visual_correct"] for o in self.val_all_outputs]) / float(sum([o["val_visual_total"] for o in self.val_all_outputs]))
        val_nonvis_acc = sum([o["val_nonvis_correct"] for o in self.val_all_outputs]) / float(sum([o["val_nonvis_total"] for o in self.val_all_outputs]))

        val_pl_acc = sum([r["val_pl_correct"] for r in n_view_res.values()]) / float(
            sum([r["val_total"] for r in n_view_res.values()])
        )
        val_pl_visual_acc = sum(
            [r["val_pl_visual_correct"] for r in n_view_res.values()]
        ) / float(sum([r["val_visual_total"] for r in n_view_res.values()]))
        val_pl_nonvis_acc = sum(
            [r["val_pl_nonvis_correct"] for r in n_view_res.values()]
        ) / float(sum([r["val_nonvis_total"] for r in n_view_res.values()]))

        val_est_err = np.mean([r["val_est_err"] for r in n_view_res.values()])

        res = {
            f"{mode}/loss": mean_val_loss,
            f"{mode}/acc": val_acc,
            f"{mode}/acc_visual": val_visual_acc,
            f"{mode}/acc_nonvis": val_nonvis_acc,
            f"{mode}/pl_acc": val_pl_acc,
            f"{mode}/pl_acc_visual": val_pl_visual_acc,
            f"{mode}/pl_acc_nonvis": val_pl_nonvis_acc,
            f"{mode}/est_err": val_est_err,
            f"{mode}/all_view_res": n_view_res,
        }

        if (
            not sanity_check
        ):  # only check best conditions and dump data if this isn't a sanity check
            if mode == "test":
                self.best_test_res = dict(res)
            else:
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_val_res = dict(res)

            dump_res = self.best_test_res if mode == "test" else self.best_val_res

            # print best result
            print("\nBest-----:")
            best_loss = dump_res[f"{mode}/loss"]
            best_acc = dump_res[f"{mode}/acc"]
            best_acc_visual = dump_res[f"{mode}/acc_visual"]
            best_acc_nonvis = dump_res[f"{mode}/acc_nonvis"]
            best_pl_acc = dump_res[f"{mode}/pl_acc"]
            best_pl_acc_visual = dump_res[f"{mode}/pl_acc_visual"]
            best_pl_acc_nonvis = dump_res[f"{mode}/pl_acc_nonvis"]
            best_est_err = dump_res[f"{mode}/est_err"]

            seed = self.cfg["train"]["random_seed"]
            json_file = os.path.join(self.save_path, f"{mode}-results-{seed}.json")
            with open(json_file, "w") as f:
                json.dump(dump_res, f, sort_keys=True, indent=4)

            print(
                f'Curr Acc: {res[f"{mode}/acc"]:0.5f} ({res[f"{mode}/pl_acc"]:0.5f}) | Visual {res[f"{mode}/acc_visual"]:0.5f} ({res[f"{mode}/pl_acc_visual"]:0.5f}) | Nonvis: {res[f"{mode}/acc_nonvis"]:0.5f} ({res[f"{mode}/pl_acc_nonvis"]:0.5f}) | Avg. Est Err: {res[f"{mode}/est_err"]:0.5f} | Val Loss: {res[f"{mode}/loss"]:0.8f} '
            )
            print(
                f"Best Acc: {best_acc:0.5f} ({best_pl_acc:0.5f}) | Visual {best_acc_visual:0.5f} ({best_pl_acc_visual:0.5f}) | Nonvis: {best_acc_nonvis:0.5f} ({best_pl_acc_nonvis:0.5f}) | Avg. Est Err: {best_est_err:0.5f} | Val Loss: {best_loss:0.8f} "
            )
            print("------------")

        if self.log_data:
            wandb.log(res)
        results = {
            "val_loss": mean_val_loss,
            "val_acc": val_acc,
            "val_visual_acc": val_visual_acc,
            "val_nonvis_acc": val_nonvis_acc,
            "val_pl_acc": val_pl_acc,
            "val_pl_visual_acc": val_pl_visual_acc,
            "val_pl_nonvis_acc": val_pl_nonvis_acc,
        }

        self.log_dict(results)
        return
