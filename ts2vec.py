# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import TensorDataset, DataLoader
# device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

# def pad_nan_to_target(array, target_length, axis=0):
#     pad_size = target_length - array.shape[axis]
#     if pad_size <= 0:
#         return array
#     npad = [(0, 0)] * array.ndim
#     npad[axis] = (0, pad_size)
#     return np.pad(array, pad_width=npad, mode='constant', constant_values=np.nan)

# def split_with_nan(x, sections, axis=0):
#     arrs = np.array_split(x, sections, axis=axis)
#     target_length = arrs[0].shape[axis]
#     for i in range(len(arrs)):
#         arrs[i] = pad_nan_to_target(arrs[i], target_length, axis=axis)
#     return arrs

# def take_per_row(A, indx, num_elem):
#     all_indx = indx[:,None] + np.arange(num_elem)
#     return A[torch.arange(all_indx.shape[0])[:,None], all_indx]

# def centerize_vary_length_series(x):
#     prefix_zeros = np.argmax(~np.isnan(x).all(axis=-1), axis=1)
#     suffix_zeros = np.argmax(~np.isnan(x[:, ::-1]).all(axis=-1), axis=1)
#     offset = (prefix_zeros + suffix_zeros) // 2 - prefix_zeros
#     rows, column_indices = np.ogrid[:x.shape[0], :x.shape[1]]
#     offset[offset < 0] += x.shape[1]
#     column_indices = column_indices - offset[:, np.newaxis]
#     return x[rows, column_indices]

# def hierarchical_contrastive_loss(z1, z2):
#     loss = torch.tensor(0., device=z1.device)
#     d = 0
#     while z1.size(1) > 1:
#         loss += .5 * instance_contrastive_loss(z1, z2)
#         loss += .5 * temporal_contrastive_loss(z1, z2)
#         z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
#         z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
#         d += 1
#     if z1.size(1) == 1:
#         loss += .5 * instance_contrastive_loss(z1, z2)
#         d += 1
#     return loss / d

# def instance_contrastive_loss(z1, z2):
#     B = z1.size(0)
#     if B == 1:
#         return z1.new_tensor(0.)
#     z = torch.cat([z1, z2], dim=0)
#     z = z.transpose(0, 1)
#     sim = torch.matmul(z, z.transpose(1, 2))
#     logits = torch.tril(sim, diagonal=-1)[:, :, :-1]
#     logits += torch.triu(sim, diagonal=1)[:, :, 1:]
#     logits = -F.log_softmax(logits, dim=-1)
#     i = torch.arange(B, device=z1.device)
#     loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
#     return loss

# def temporal_contrastive_loss(z1, z2):
#     T = z1.size(1)
#     if T == 1:
#         return z1.new_tensor(0.)
#     z = torch.cat([z1, z2], dim=1)
#     sim = torch.matmul(z, z.transpose(1, 2))
#     logits = torch.tril(sim, diagonal=-1)[:, :, :-1]
#     logits += torch.triu(sim, diagonal=1)[:, :, 1:]
#     logits = -F.log_softmax(logits, dim=-1)
#     t = torch.arange(T, device=z1.device)
#     loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
#     return loss

# class SamePadConv(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, dilation):
#         super().__init__()
#         self.receptive_field = (kernel_size - 1) * dilation + 1
#         padding = self.receptive_field // 2
#         self.conv = nn.Conv1d(
#             in_channels, out_channels, kernel_size,
#             padding=padding,
#             dilation=dilation,
#             groups=1)
#         self.remove = 1 if self.receptive_field % 2 == 0 else 0

#     def forward(self, x):
#         out = self.conv(x)
#         if self.remove > 0:
#             out = out[:, :, : -self.remove]
#         return out

# class ConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
#         super().__init__()
#         self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation)
#         self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation)
#         self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None

#     def forward(self, x):
#         residual = x if self.projector is None else self.projector(x)
#         x = F.gelu(x)
#         x = self.conv1(x)
#         x = F.gelu(x)
#         x = self.conv2(x)
#         return x + residual

# class DilatedConvEncoder(nn.Module):
#     def __init__(self, in_channels, channels, kernel_size=3):
#         super().__init__()
#         self.net = nn.Sequential(*[
#             ConvBlock(
#                 channels[i-1] if i > 0 else in_channels,
#                 channels[i],
#                 kernel_size=kernel_size,
#                 dilation=2**i,
#                 final=(i == len(channels)-1))
#             for i in range(len(channels))])

#     def forward(self, x):
#         return self.net(x)

# class TSEncoder(nn.Module):
#     def __init__(self, input_dims, output_dims, hidden_dims=64, depth=10):
#         super().__init__()
#         self.input_dims = input_dims
#         self.output_dims = output_dims
#         self.hidden_dims = hidden_dims
#         self.input_fc = nn.Linear(input_dims, hidden_dims)
#         self.feature_extractor = DilatedConvEncoder(
#             hidden_dims,
#             [hidden_dims] * depth + [output_dims])
#         self.repr_dropout = nn.Dropout(p=0.1)

#     def forward(self, x): 
#         nan_mask = ~x.isnan().any(axis=-1)
#         x[~nan_mask] = 0
#         x = self.input_fc(x)
#         if self.training:
#             mask = torch.from_numpy(np.random.binomial(1, .5, size=(x.size(0), x.size(1)))).to(torch.bool).to(x.device)
#         else:
#             mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
#         mask &= nan_mask
#         x[~mask] = 0
#         x = x.transpose(1, 2) 
#         x = self.repr_dropout(self.feature_extractor(x)) 
#         x = x.transpose(1, 2) 
#         return x

# class TS2Vec:
#     def __init__(self, input_dims, output_dims=320, hidden_dims=64, depth=10, device='cuda'):
#         super().__init__()
#         self._net = TSEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth).to(device)
#         self.net = torch.optim.swa_utils.AveragedModel(self._net)
#         self.net.update_parameters(self._net)
#         self.n_epochs = 0
#         self.n_iters = 0

#     def fit(self, train_data, n_epochs=20):
#         temporal_missing = np.isnan(train_data).all(axis=-1).any(axis=0)
#         if temporal_missing[0] or temporal_missing[-1]:
#             train_data = centerize_vary_length_series(train_data)
#         train_data = train_data[~np.isnan(train_data).all(axis=2).all(axis=1)]
#         train_dataset = TensorDataset(torch.from_numpy(train_data).to(torch.float))
#         train_loader = DataLoader(train_dataset, batch_size=min(len(train_dataset), 16), shuffle=True, drop_last=True)
#         optimizer = torch.optim.AdamW(self._net.parameters(), lr=.001)
#         loss_log = []
#         while True:
#             if self.n_epochs >= n_epochs:
#                 break
#             cum_loss = 0
#             n_epoch_iters = 0
#             for batch in train_loader:
#                 x = batch[0]
#                 x = x.to(device)
#                 ts_l = x.size(1)
#                 crop_l = np.random.randint(low=2, high=ts_l+1)
#                 crop_left = np.random.randint(ts_l - crop_l + 1)
#                 crop_right = crop_left + crop_l
#                 crop_eleft = np.random.randint(crop_left + 1)
#                 crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
#                 crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))
#                 optimizer.zero_grad()
#                 out1 = self._net(take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft))
#                 out1 = out1[:, -crop_l:]
#                 out2 = self._net(take_per_row(x, crop_offset + crop_left, crop_eright - crop_left))
#                 out2 = out2[:, :crop_l]
#                 loss = hierarchical_contrastive_loss(out1, out2)
#                 loss.backward()
#                 optimizer.step()
#                 self.net.update_parameters(self._net)
#                 cum_loss += loss.item()
#                 n_epoch_iters += 1
#                 self.n_iters += 1
#             cum_loss /= n_epoch_iters
#             loss_log.append(cum_loss)
#             print(f"Epoch #{self.n_epochs}: loss={cum_loss}")
#             self.n_epochs += 1
#         return loss_log

#     def _eval_with_pooling(self, x):
#         out = self.net(x.to(device, non_blocking=True), None)
#         out = F.max_pool1d(
#             out.transpose(1, 2),
#             kernel_size = out.size(1),
#         ).transpose(1, 2)
#         return out.cpu()

#     def encode(self, data):
#         org_training = self.net.training
#         self.net.eval()
#         dataset = TensorDataset(torch.from_numpy(data).to(torch.float))
#         loader = DataLoader(dataset, batch_size=16)
#         with torch.no_grad():
#             output = []
#             for batch in loader:
#                 x = batch[0]
#                 out = self._eval_with_pooling(x)
#                 out = out.squeeze(1)
#                 output.append(out)
#             output = torch.cat(output, dim=0)
#         self.net.train(org_training)
#         return output.numpy()
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

def pad_nan_to_target(array, target_length, axis=0):
    pad_size = target_length - array.shape[axis]
    if pad_size <= 0:
        return array
    npad = [(0, 0)] * array.ndim
    npad[axis] = (0, pad_size)
    return np.pad(array, pad_width=npad, mode='constant', constant_values=np.nan)

def split_with_nan(x, sections, axis=0):
    arrs = np.array_split(x, sections, axis=axis)
    target_length = arrs[0].shape[axis]
    for i in range(len(arrs)):
        arrs[i] = pad_nan_to_target(arrs[i], target_length, axis=axis)
    return arrs

def take_per_row(A, indx, num_elem):
    all_indx = indx[:,None] + np.arange(num_elem)
    return A[torch.arange(all_indx.shape[0])[:,None], all_indx]

def centerize_vary_length_series(x):
    prefix_zeros = np.argmax(~np.isnan(x).all(axis=-1), axis=1)
    suffix_zeros = np.argmax(~np.isnan(x[:, ::-1]).all(axis=-1), axis=1)
    offset = (prefix_zeros + suffix_zeros) // 2 - prefix_zeros
    rows, column_indices = np.ogrid[:x.shape[0], :x.shape[1]]
    offset[offset < 0] += x.shape[1]
    column_indices = column_indices - offset[:, np.newaxis]
    return x[rows, column_indices]

def hierarchical_contrastive_loss(z1, z2):
    loss = torch.tensor(0., device=z1.device)
    d = 0
    while z1.size(1) > 1:
        loss += .5 * instance_contrastive_loss(z1, z2)
        loss += .5 * temporal_contrastive_loss(z1, z2)
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
        d += 1
    if z1.size(1) == 1:
        loss += .5 * instance_contrastive_loss(z1, z2)
        d += 1
    return loss / d

def instance_contrastive_loss(z1, z2):
    B = z1.size(0)
    if B == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=0)
    z = z.transpose(0, 1)
    sim = torch.matmul(z, z.transpose(1, 2))
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    i = torch.arange(B, device=z1.device)
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    return loss

def temporal_contrastive_loss(z1, z2):
    T = z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=1)
    sim = torch.matmul(z, z.transpose(1, 2))
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    t = torch.arange(T, device=z1.device)
    loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    return loss

class SamePadConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding,
            dilation=dilation,
            groups=1)
        self.remove = 1 if self.receptive_field % 2 == 0 else 0

    def forward(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation)
        self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None

    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual

class DilatedConvEncoder(nn.Module):
    def __init__(self, in_channels, channels, kernel_size=3):
        super().__init__()
        self.net = nn.Sequential(*[
            ConvBlock(
                channels[i-1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                dilation=2**i,
                final=(i == len(channels)-1))
            for i in range(len(channels))])

    def forward(self, x):
        return self.net(x)

class TSEncoder(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=10):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims])
        self.repr_dropout = nn.Dropout(p=0.1)

    def forward(self, x): 
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        x = self.input_fc(x)
        if self.training:
            mask = torch.from_numpy(np.random.binomial(1, .5, size=(x.size(0), x.size(1)))).to(torch.bool).to(x.device)
        else:
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        mask &= nan_mask
        x[~mask] = 0
        x = x.transpose(1, 2) 
        x = self.repr_dropout(self.feature_extractor(x)) 
        x = x.transpose(1, 2) 
        return x

class TS2Vec:
    def __init__(self, input_dims, device, output_dims=320, hidden_dims=64, depth=10):
        super().__init__()
        self._net = TSEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth).to(device)
        self.net = torch.optim.swa_utils.AveragedModel(self._net)
        self.net.update_parameters(self._net)
        self.n_epochs = 0
        self.n_iters = 0
        self.device = device

    def fit(self, train_data, n_epochs=20):
        temporal_missing = np.isnan(train_data).all(axis=-1).any(axis=0)
        if temporal_missing[0] or temporal_missing[-1]:
            train_data = centerize_vary_length_series(train_data)
        train_data = train_data[~np.isnan(train_data).all(axis=2).all(axis=1)]
        train_dataset = TensorDataset(torch.from_numpy(train_data).to(torch.float))
        train_loader = DataLoader(train_dataset, batch_size=min(len(train_dataset), 16), shuffle=True, drop_last=True)
        optimizer = torch.optim.AdamW(self._net.parameters(), lr=.001)
        loss_log = []
        while True:
            if self.n_epochs >= n_epochs:
                break
            cum_loss = 0
            n_epoch_iters = 0
            for batch in train_loader:
                x = batch[0]
                x = x.to(self.device)
                ts_l = x.size(1)
                crop_l = np.random.randint(low=2, high=ts_l+1)
                crop_left = np.random.randint(ts_l - crop_l + 1)
                crop_right = crop_left + crop_l
                crop_eleft = np.random.randint(crop_left + 1)
                crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
                crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))
                optimizer.zero_grad()
                out1 = self._net(take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft))
                out1 = out1[:, -crop_l:]
                out2 = self._net(take_per_row(x, crop_offset + crop_left, crop_eright - crop_left))
                out2 = out2[:, :crop_l]
                loss = hierarchical_contrastive_loss(out1, out2)
                loss.backward()
                optimizer.step()
                self.net.update_parameters(self._net)
                cum_loss += loss.item()
                n_epoch_iters += 1
                self.n_iters += 1
            cum_loss /= n_epoch_iters
            loss_log.append(cum_loss)
            print(f"Epoch #{self.n_epochs}: loss={cum_loss}")
            self.n_epochs += 1
        return loss_log

    def _eval_with_pooling(self, x):
        out = self.net(x.to(self.device, non_blocking=True), None)
        out = F.max_pool1d(
            out.transpose(1, 2),
            kernel_size = out.size(1),
        ).transpose(1, 2)
        return out.cpu()

    def encode(self, data):
        org_training = self.net.training
        self.net.eval()
        dataset = TensorDataset(torch.from_numpy(data).to(torch.float))
        loader = DataLoader(dataset, batch_size=16)
        with torch.no_grad():
            output = []
            for batch in loader:
                x = batch[0]
                out = self._eval_with_pooling(x)
                out = out.squeeze(1)
                output.append(out)
            output = torch.cat(output, dim=0)
        self.net.train(org_training)
        return output.numpy()
