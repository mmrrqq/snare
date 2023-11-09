from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.rotator import Rotator


class PredictiveRotator(Rotator):
    caption_fc: nn.Module

    def __init__(self, cfg, train_ds, val_ds, freeze_mapping_layer=True):
        self.estimate_init_state = False
        self.estimate_final_state = False
        super().__init__(cfg, train_ds, val_ds, freeze_mapping_layer)

    def build_model(self):
        # caption to view prediction
        self.caption_fc = nn.Sequential(
            nn.Linear(self.lang_feat_dim, 512),
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
        ((img1_n_feats, img2_n_feats), lang_feats, ans, _, _, is_visual) = batch
        outputs = {}

        # to device
        img1_n_feats = img1_n_feats.to(device=self.device).float()
        img2_n_feats = img2_n_feats.to(device=self.device).float()
        lang_feats = lang_feats.to(device=self.device).float()

        # perform mapping
        lang_feats = self.lang_fc(lang_feats)
        img1_n_feats = self.img_fc(img1_n_feats)
        img2_n_feats = self.img_fc(img2_n_feats)

        # estimate current view
        init_state_estimation = self.estimate_view(
            img1_n_feats,
            img2_n_feats,
            lang_feats,
            ans,
            view_override,
        )

        # output variables from state estimation
        bs = img1_n_feats.shape[0]

        init_views1 = init_state_estimation["views1"]
        init_views2 = init_state_estimation["views2"]

        est_init_views1 = init_state_estimation["est_views1"]
        est_init_views2 = init_state_estimation["est_views2"]

        loss = (
            init_state_estimation["loss"]
            if self.estimate_init_state
            else torch.tensor(0)
        )

        if view_override is not None:
            (
                img1_chosen_feats,
                img2_chosen_feats,
                rotated_views1,
                rotated_views2,
            ) = self.choose_feats_from_random_views(
                bs, img1_n_feats, img2_n_feats, init_views1, init_views2, view_override
            )
        else:
            est_best_caption_view = init_state_estimation["caption_view_est"]
            rotated_views1 = rotated_views2 = est_best_caption_view.int()
            if self.training:
                init_best_views1 = init_state_estimation["best_views_model1"]
                init_best_views2 = init_state_estimation["best_views_model2"]
                init_best_views = torch.tensor(
                    [
                        init_best_views1[i] if ans[i] < 1 else init_best_views2[i]
                        for i in range(bs)
                    ]
                ).to(self.device)
                total_correct_best_caption_view_est = int(
                    torch.count_nonzero(est_best_caption_view == init_best_views)
                )
                caption_view_prediction_error = (
                    self.modulo_views(init_best_views - est_best_caption_view)
                    .abs()
                    .float()
                    .mean()
                )
                outputs = outputs | {
                    "caption_view_prediction_error": caption_view_prediction_error,
                    "total_correct_best_view_est": total_correct_best_caption_view_est,
                }
                
            outputs = outputs | {
                "init_est_caption_view_loss": init_state_estimation[
                    "est_caption_view_loss"
                ]
                if self.training
                else 0,
            }

            img1_chosen_feats = torch.stack(
                [img1_n_feats[i, rotated_views1[i], :] for i in range(bs)]
            )
            img2_chosen_feats = torch.stack(
                [img2_n_feats[i, rotated_views2[i], :] for i in range(bs)]
            )

        # estimate second view before performing prediction
        final_state_estimation = self.estimate_view(
            img1_n_feats,
            img2_n_feats,
            lang_feats,
            ans,
        )
        est_final_views1 = final_state_estimation["est_views1"]
        est_final_views2 = final_state_estimation["est_views2"]

        if self.estimate_final_state:
            loss += final_state_estimation["loss"]

        # classifier probablities chosen features
        img1_chosen_prob = self.cls_fc(
            torch.cat([img1_chosen_feats, lang_feats], dim=-1)
        )
        img2_chosen_prob = self.cls_fc(
            torch.cat([img2_chosen_feats, lang_feats], dim=-1)
        )

        # classifier loss
        raw_probs = torch.cat([img1_chosen_prob, img2_chosen_prob], dim=-1)
        probs = F.softmax(raw_probs, dim=-1)
        num_steps = torch.ones((bs)).to(dtype=torch.long, device=lang_feats.device) * 2

        test_mode = ans[0] == -1
        outputs = outputs | {
            "probs": probs,
            "num_steps": num_steps,
        }
        if not test_mode:
            # classifier loss
            cls_labels = F.one_hot(ans, num_classes=2)
            cls_loss_weight = self.cfg["train"]["loss"]["cls_weight"]
            loss += (
                self.smoothed_cross_entropy(raw_probs, cls_labels)
            ) * cls_loss_weight            

            return outputs | {
                "probs": probs,
                "action_loss": loss,
                "labels": cls_labels,
                "is_visual": is_visual,
                "num_steps": num_steps,                
                "est_init_views1": est_init_views1,
                "est_init_views2": est_init_views2,
                "est_final_views1": est_final_views1,
                "est_final_views2": est_final_views2,
            }
        else:
            return outputs

    def forward_all_views(
        self,
        lang_feats: torch.Tensor,
        model1_feats: torch.Tensor,
        model2_feats: torch.Tensor,
    ):
        # create view on lang_feat tensor: repeat same tensor of size L for all views V -> (B, V, L)
        lang_feats_view = lang_feats.expand(self.num_views, -1, -1).permute(1, 0, 2)

        with torch.no_grad():
            all_probs1 = self.cls_fc(torch.cat([model1_feats, lang_feats_view], dim=-1))
            all_probs2 = self.cls_fc(torch.cat([model2_feats, lang_feats_view], dim=-1))

        all_probs = torch.stack([all_probs1, all_probs2], dim=-2).squeeze()
        all_probs = F.softmax(all_probs, dim=2)

        return all_probs

    def get_best_views(self, all_view_probs: torch.Tensor, gt_model_ref: torch.Tensor):
        # best views with highest classifier probs
        best_views_model1 = all_view_probs[:, :, 0].argmax(-1)
        best_views_model2 = all_view_probs[:, :, 1].argmax(-1)

        # this is required for caption view prediction
        best_views_per_expression = torch.tensor(
            [
                best_views_model1[i] if gt_model_ref[i] < 1 else best_views_model2[i]
                for i in range(best_views_model1.shape[0])
            ]
        ).to(self.device)
        gt_best_view = F.one_hot(
            best_views_per_expression.to(torch.int64), num_classes=self.num_views
        )

        return gt_best_view, best_views_model1, best_views_model2

    def estimate_view(
        self,
        model1_n_feats: torch.Tensor,
        model2_n_feats: torch.Tensor,
        lang_feats: torch.Tensor,
        gt_img_id: torch.Tensor,
        view_override: Optional[int | torch.Tensor] = None,
    ):
        outputs = super().estimate_view(
            model1_n_feats, model2_n_feats, lang_feats, view_override
        )

        if view_override is None:
            est_caption_view_logits = self.caption_fc(lang_feats)

            # state estimation loss
            if self.training:
                all_view_probs = self.forward_all_views(
                    lang_feats, model1_n_feats, model2_n_feats
                )
                (
                    gt_best_views,
                    best_views_model1,
                    best_views_model2,
                ) = self.get_best_views(all_view_probs, gt_img_id)
                outputs["best_views_model1"] = best_views_model1
                outputs["best_views_model2"] = best_views_model2

                caption_view_est_weight = self.cfg["train"]["loss"][
                    "caption_view_est_weight"
                ]

                # only add caption view estimation after 30 batches..
                caption_estimation_loss = (
                    self.smoothed_cross_entropy(est_caption_view_logits, gt_best_views)
                    * caption_view_est_weight
                )
                outputs["loss"] += caption_estimation_loss
                outputs["est_caption_view_loss"] = (caption_estimation_loss,)

            caption_view_est = F.softmax(est_caption_view_logits, dim=-1).argmax(-1)

            outputs["caption_view_est"] = caption_view_est

        return outputs
