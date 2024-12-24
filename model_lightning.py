import torch
import torch.nn.functional as F
import lightning as L



class LitNetwork(L.LightningModule):
    def __init__(self, network):
        super().__init__()
        self.network = network
        self.save_hyperparameters()
    
    def training_step(self,batch, batch_idx):
        past_xyz_points_batch, past_t_index_batch, occupied_points_batch, unoccupied_points_batch = batch
        occupied_preds, unoccupied_preds = self.network(
            [points.cuda() for points in past_xyz_points_batch],
            [t_idx.cuda() for t_idx in past_t_index_batch],
            occupied_points_batch = occupied_points_batch.cuda(),
            unoccupied_points_batch = unoccupied_points_batch.cuda(),
        )
        loss = (
            F.binary_cross_entropy_with_logits(occupied_preds, torch.ones_like(occupied_preds))
            + F.binary_cross_entropy_with_logits(unoccupied_preds, torch.zeros_like(unoccupied_preds))
        ) / 2
        occupied_correct = torch.sum(occupied_preds >= 0.5).item()
        unoccupied_correct = torch.sum(unoccupied_preds < 0.5).item()
        occupied_correct /= occupied_preds.shape[0] * occupied_preds.shape[1]
        unoccupied_correct /= unoccupied_preds.shape[0] * unoccupied_preds.shape[1]
        
       
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('Occupied Correct', occupied_correct, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log('Unoccupied Correct', unoccupied_correct, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.logger.experiment.add_scalars('Occupied Correct/Unoccupied Correct Training', 
                      {'Occupied' : occupied_correct.item() * 100,
                       'Unoccupied': unoccupied_correct.item() * 100})
        self.logger.experiment.add_scalar("Losses", {'Training Loss': loss}, self.global_step)
        return loss
    def validation_step(self, batch, batch_idx):
        past_xyz_points_batch, past_t_index_batch, occupied_points_batch, unoccupied_points_batch = batch
        occupied_preds, unoccupied_preds = self.network(
            [points.cuda() for points in past_xyz_points_batch],
            [t_idx.cuda() for t_idx in past_t_index_batch],
            [occ.cuda() for occ in occupied_points_batch],
            [unocc.cuda() for unocc in unoccupied_points_batch],
        )
        loss = (
            F.binary_cross_entropy_with_logits(occupied_preds, torch.ones_like(occupied_preds))
            + F.binary_cross_entropy_with_logits(unoccupied_preds, torch.zeros_like(unoccupied_preds))
        ) / 2
        occupied_correct = torch.sum(occupied_preds >= 0.5).item()        
        unoccupied_correct = torch.sum(unoccupied_preds < 0.5).item()
        

        occupied_correct /= occupied_preds.shape[0] * occupied_preds.shape[1] 
        unoccupied_correct /= unoccupied_preds.shape[0] * unoccupied_preds.shape[1]
        
        """
        When you call self.log inside the validation_step and test_step, 
        Lightning automatically accumulates the metric and averages it once it’s gone through the whole split (epoch).
        """
        self.logger.experiment.add_scalars('Occupied Correct/Unoccupied Correct Validation', 
                      {'Occupied' : occupied_correct.item() * 100,
                       'Unoccupied': unoccupied_correct.item() * 100})
        self.logger.experiment.add_scalar("Losses", {'Validation Loss': loss}, self.global_step)
        value = {"Validation Loss": loss, "Occupied Correct": occupied_correct,  
            "Unoccupied Correct": unoccupied_correct}
        return value
        

    def configure_optimizers(self):
        target_lr=8.0e-4
        total_iters=800_000
        warmup_iters=1_000
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=8e-5, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=target_lr,
            total_steps=total_iters,
            anneal_strategy="cos",
            pct_start=warmup_iters / total_iters,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
