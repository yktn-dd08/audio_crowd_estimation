import torch
import torch.nn as nn
import torchaudio
from transformers import ASTFeatureExtractor, ASTForAudioClassification


class ASTRegressor(nn.Module):
    def __init__(self, feat_num=256, drop_out=0.3, out_features=1,
                 model_name='MIT/ast-finetuned-audioset-10-10-0.4593', finetune=False):
        super(ASTRegressor, self).__init__()
        self.feature_extractor = ASTFeatureExtractor.from_pretrained(model_name)
        self.ast_model = ASTForAudioClassification.from_pretrained(model_name)
        if not finetune:
            for param in self.ast_model.base_model.parameters():
                param.requires_grad = False
        self.regressor = nn.Sequential(
            nn.Linear(in_features=self.ast_model.config.hidden_size, out_features=feat_num),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(in_features=feat_num, out_features=out_features)
        )

    def forward(self, x):
        x = self.ast_model(x, output_hidden_states=True)
        x = x.hidden_states[-1][:, 0, :]
        x = self.regressor(x)
        return x