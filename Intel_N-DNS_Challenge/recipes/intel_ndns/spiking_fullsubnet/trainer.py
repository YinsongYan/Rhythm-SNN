import torch
from pathlib import Path
import pandas as pd
from accelerate.logging import get_logger
from tqdm import tqdm

from audiozen.acoustics.audio_feature import save_wav
from audiozen.loss import SISNRLoss, freq_MAE, mag_MAE
from audiozen.metric import DNSMOS, PESQ, SISDR, STOI, compute_synops, compute_neuronops
from audiozen.trainer import Trainer as BaseTrainer


logger = get_logger(__name__)


class Trainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dns_mos = DNSMOS(input_sr=self.sr, device=self.accelerator.process_index)
        self.stoi = STOI(sr=self.sr)
        self.pesq_wb = PESQ(sr=self.sr, mode="wb")
        self.pesq_nb = PESQ(sr=self.sr, mode="nb")
        self.sisnr_loss = SISNRLoss(return_neg=False)
        self.si_sdr = SISDR()
        self.north_star_metric = "si_sdr"

    def training_step(self, batch, batch_idx):
        self.optimizer.zero_grad()

        noisy_y, clean_y, _ = batch

        batch_size, *_ = noisy_y.shape

        enhanced_y, enhanced_mag, *_ = self.model(noisy_y)

        loss_freq_mae = freq_MAE(enhanced_y, clean_y)
        loss_mag_mae = mag_MAE(enhanced_y, clean_y)
        loss_sdr = self.sisnr_loss(enhanced_y, clean_y)
        loss_sdr_norm = 0.001 * (100 - loss_sdr)
        loss = loss_freq_mae + loss_mag_mae + loss_sdr_norm  # + loss_g_fake

        self.accelerator.backward(loss)
        self.optimizer.step()

        return {
            "loss": loss,
            "loss_freq_mae": loss_freq_mae,
            "loss_mag_mae": loss_mag_mae,
            "loss_sdr": loss_sdr,
            "loss_sdr_norm": loss_sdr_norm,
        }

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # noisy_y, clean_y, noisy_file = batch
        # enhanced_y, *_ = self.model(noisy_y)
        mix_y, ref_y, id = batch
        # est_y, *_ = self.model(mix_y)
        est_y, enhanced_mag, fb_out, sb_out = self.model(mix_y)


        if len(id) != 1:
            raise ValueError(f"Expected batch size 1 during validation, got {len(id)}")

        # calculate metrics
        mix_y = mix_y.squeeze(0).detach().cpu().numpy()
        ref_y = ref_y.squeeze(0).detach().cpu().numpy()
        est_y = est_y.squeeze(0).detach().cpu().numpy()

        # caculate
        # synops = compute_synops(fb_out, sb_out, shared_weights=self.config["model"]["args"]["shared_weights"])
        # neuron_ops = compute_neuronops(fb_out, sb_out)

        # # to tensor
        # synops = torch.tensor([synops], device=self.accelerator.device).unsqueeze(0)
        # synops = synops.repeat(est_y.shape[0], 1)
        # neuron_ops = torch.tensor([neuron_ops], device=self.accelerator.device).unsqueeze(0)
        # neuron_ops = neuron_ops.repeat(est_y.shape[0], 1)
        # print("synops:", synops)
        # print("neuron_ops:", neuron_ops)

        # # save enhanced audio
        stem = Path(id[0]).stem
        enhanced_dir = self.enhanced_dir / f"dataloader_{dataloader_idx}"
        enhanced_dir.mkdir(exist_ok=True, parents=True)
        enhanced_fpath = enhanced_dir / f"{stem}_raw.wav"
        est_fpath = enhanced_dir / f"{stem}_est.wav"
        ref_fpath = enhanced_dir / f"{stem}_ref.wav"
        save_wav(mix_y, enhanced_fpath.as_posix(), self.sr)
        save_wav(est_y, est_fpath.as_posix(), self.sr)
        save_wav(ref_y, ref_fpath.as_posix(), self.sr)

        si_sdr = self.si_sdr(est_y, ref_y)
        dns_mos = self.dns_mos(est_y)
        
        # out = si_sdr | dns_mos | {"synops": synops.item()} | {"neuron_ops": neuron_ops.item()}
        # print the synops and neuron_ops
        # out = si_sdr | dns_mos | {"synops": synops} | {"neuron_ops": neuron_ops}
        out = si_sdr | dns_mos
        return [out]

    def validation_epoch_end(self, outputs, log_to_tensorboard=True):
        score = 0.0

        for dataloader_idx, dataloader_outputs in enumerate(outputs):
            logger.info(f"Computing metrics on epoch {self.state.epochs_trained} for dataloader {dataloader_idx}...")

            loss_dict_list = []
            for step_loss_dict_list in tqdm(dataloader_outputs):
                loss_dict_list.extend(step_loss_dict_list)

            df_metrics = pd.DataFrame(loss_dict_list)

            # Compute mean of all metrics
            df_metrics_mean = df_metrics.mean(numeric_only=True)
            df_metrics_mean_df = df_metrics_mean.to_frame().T

            time_now = self._get_time_now()
            df_metrics.to_csv(
                self.metrics_dir / f"dl_{dataloader_idx}_epoch_{self.state.epochs_trained}_{time_now}.csv",
                index=False,
            )
            df_metrics_mean_df.to_csv(
                self.metrics_dir / f"dl_{dataloader_idx}_epoch_{self.state.epochs_trained}_{time_now}_mean.csv",
                index=False,
            )

            logger.info(f"\n{df_metrics_mean_df.to_markdown()}")
            score += df_metrics_mean[self.north_star_metric]

            if log_to_tensorboard:
                for metric, value in df_metrics_mean.items():
                    self.writer.add_scalar(f"metrics_{dataloader_idx}/{metric}", value, self.state.epochs_trained)

        return score

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.validation_step(batch, batch_idx, dataloader_idx)

    def test_epoch_end(self, outputs, log_to_tensorboard=True):
        return self.validation_epoch_end(outputs, log_to_tensorboard=False)
