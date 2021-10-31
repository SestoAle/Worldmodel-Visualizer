import torch
import torch.nn as nn
import random

class model_encdec(nn.Module):
    """
    Encoder-Decoder model. The model reconstructs the future trajectory from an encoding of both past and future.
    Past and future trajectories are encoded separately.
    A trajectory is first convolved with a 1D kernel and are then encoded with a Gated Recurrent Unit (GRU).
    Encoded states are concatenated and decoded with a GRU and a fully connected layer.
    The decoding process decodes the trajectory step by step, predicting offsets to be added to the previous point.
    """
    def __init__(self, settings):
        super(model_encdec, self).__init__()

        self.name_model = 'autoencoder'
        self.use_cuda = settings["use_cuda"]
        self.dim_embedding_key = settings["dim_embedding_key"]
        self.traj_len = settings["past_len"]
        self.channel_in = 3
        # channel_in = 6
        channel_out = 16
        dim_kernel = 3
        input_gru = channel_out

        # # temporal encoding
        self.conv_past_encoder = nn.Conv1d(self.channel_in, channel_out, dim_kernel, stride=1, padding=1)

        # encoder-decoder
        self.encoder_past = nn.LSTM(input_gru, self.dim_embedding_key, 1, batch_first=True)

        # temporal encoding
        self.conv_past_decoder = nn.Conv1d(self.channel_in, channel_out, dim_kernel, stride=1, padding=1)
        # self.encoder_past = nn.GRU(channel_in, self.dim_embedding_key, 1, batch_first=True)
        self.decoder = nn.LSTM(self.channel_in, self.dim_embedding_key, 1, batch_first=True)
        self.FC_output = torch.nn.Linear(self.dim_embedding_key, self.channel_in)

        # activation function
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        # weight initialization: kaiming
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.conv_past_encoder.weight)
        nn.init.kaiming_normal_(self.conv_past_decoder.weight)
        nn.init.kaiming_normal_(self.encoder_past.weight_ih_l0)
        nn.init.kaiming_normal_(self.encoder_past.weight_hh_l0)
        nn.init.kaiming_normal_(self.decoder.weight_ih_l0)
        nn.init.kaiming_normal_(self.decoder.weight_hh_l0)
        nn.init.kaiming_normal_(self.FC_output.weight)

        nn.init.zeros_(self.conv_past_encoder.bias)
        nn.init.zeros_(self.conv_past_decoder.bias)
        nn.init.zeros_(self.encoder_past.bias_ih_l0)
        nn.init.zeros_(self.encoder_past.bias_hh_l0)
        nn.init.zeros_(self.decoder.bias_ih_l0)
        nn.init.zeros_(self.decoder.bias_hh_l0)
        nn.init.zeros_(self.FC_output.bias)

    def forward(self, trajectories, teacher=0.5):
        """
        Forward pass that encodes past and future and decodes the future.
        :param trajectories: past trajectory
        :param future: future trajectory
        :return: decoded future
        """

        teacher_forcing_ratio = teacher

        teacher_force = random.random() < teacher_forcing_ratio

        dim_batch = trajectories.size()[0]
        prediction = torch.Tensor()
        zero_padding = torch.zeros(dim_batch, 1, self.channel_in)
        if self.use_cuda:
            zero_padding = zero_padding.cuda()
            prediction = prediction.cuda()

        # temporal encoding for the original trajectory
        trajectories = torch.transpose(trajectories, 1, 2)
        traj_embed = self.relu(self.conv_past_encoder(trajectories))
        traj_embed = torch.transpose(traj_embed, 1, 2)

        # sequence encoding
        output_emb, state_emb = self.encoder_past(traj_embed)

        # state concatenation and decoding
        input_fut = zero_padding
        state_fut = state_emb[0]
        c_fut = state_emb[1]
        input_fut = torch.transpose(input_fut, 1, 2)
        for i in range(self.traj_len):
            #input_fut = self.relu(self.conv_past_decoder(input_fut))
            input_fut = torch.transpose(input_fut, 1, 2)
            output_decoder, (state_fut, c_fut) = self.decoder(input_fut, (state_fut, c_fut))
            coords_next = self.FC_output(output_decoder)
            prediction = torch.cat((prediction, coords_next), 1)

            if teacher_force and i < self.traj_len - 1:
                input_fut = trajectories[:, :, i + 1]
                input_fut = torch.unsqueeze(input_fut, 2)
            else:
                input_fut = coords_next
                input_fut = torch.transpose(input_fut, 1, 2)

        return prediction, state_emb
