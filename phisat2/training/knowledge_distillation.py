import lightning as L
import torch
import torch.nn as nn
from typing import Any

class KDModule(L.LightningModule):
    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        spec: Any,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["student_model", "teacher_model"])
        
        self.student = student_model
        self.teacher = teacher_model
        
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Any:
        return self.student(x)

    def _dummy_loss(self, student_features: Any) -> torch.Tensor:
        
        loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        if isinstance(student_features, dict):
            for v in student_features.values():
                if v is not None:
                    loss = loss + v.mean() * 0.0
        elif isinstance(student_features, (list, tuple)):
            for v in student_features:
                loss = loss + v.mean() * 0.0
        else:
            loss = loss + student_features.mean() * 0.0
            
        return loss + 1.0

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        images_teacher = batch["sentinel2"]
        images_student = batch["simulated"]
        
        with torch.no_grad():
            self.teacher.eval()
            teacher_features = self.teacher(images_teacher)
        
        student_features = self.student(images_student)
        
        print("Teacher features:")
        for f in teacher_features:
            print(f.shape)
        
        print("Student features:")
        for f in student_features:
            print(f.shape)
        
        #interrupt for testing (breakpoint)
        print("interrupt for testing: training_step executed")
        breakpoint()
        
        loss = self._dummy_loss(student_features)
        
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        images_teacher = batch["sentinel2"]
        images_student = batch["simulated"]

        with torch.no_grad():
            teacher_features = self.teacher(images_teacher)
            student_features = self.student(images_student)
            
        loss = self._dummy_loss(student_features)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.student.parameters()),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer