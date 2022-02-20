from unet_ddpm import *

class EncResBlock(nn.Module):
    def __init__(
            self, in_channel, out_channel, dropout=0, group_norm=32,
    ):
        super().__init__()

        norm_affine = True

        self.norm1 = nn.GroupNorm(group_norm, in_channel)
        self.activation1 = Swish()
        self.conv1 = conv2d(in_channel, out_channel, 3, padding=1)

        self.norm2 = nn.GroupNorm(group_norm, out_channel, affine=norm_affine)
        self.activation2 = Swish()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = conv2d(out_channel, out_channel, 3, padding=1, scale=1e-10)

        if in_channel != out_channel:
            self.skip = conv2d(in_channel, out_channel, 1)

        else:
            self.skip = None

    def forward(self, input):
        batch = input.shape[0]

        out = self.conv1(self.activation1(self.norm1(input)))

        out = self.norm2(out)

        out = self.conv2(self.dropout(self.activation2(out)))

        if self.skip is not None:
            input = self.skip(input)

        return out + input

class EncResBlockWithAttention(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            dropout,
            use_attention=False,
            attention_head=1,
            group_norm=32,
    ):
        super().__init__()

        self.resblocks = EncResBlock(
            in_channel, out_channel, dropout, group_norm=group_norm
        )

        if use_attention:
            self.attention = SelfAttention(out_channel, n_head=attention_head, group_norm=group_norm)

        else:
            self.attention = None

    def forward(self, input):
        out = self.resblocks(input)

        if self.attention is not None:
            out = self.attention(out)

        return out

class BaseEncoder(nn.Module):
    def __init__(
            self,
            in_channel: StrictInt,
            channel: StrictInt,
            channel_multiplier: List[StrictInt],
            n_res_blocks: StrictInt,
            attn_strides: List[StrictInt],
            attn_heads: StrictInt = 1,
            dropout: StrictFloat = 0,
            fold: StrictInt = 1,
    ):
        super().__init__()

        self.fold = fold

        group_norm = channel // 4

        n_block = len(channel_multiplier)

        down_layers = [conv2d(in_channel * (fold ** 2), channel, 3, padding=1)]
        feat_channels = [channel]
        in_channel = channel
        for i in range(n_block):
            for _ in range(n_res_blocks):
                channel_mult = channel * channel_multiplier[i]

                down_layers.append(
                    EncResBlockWithAttention(
                        in_channel,
                        channel_mult,
                        dropout,
                        use_attention=2 ** i in attn_strides,
                        attention_head=attn_heads,
                        group_norm=group_norm
                    )
                )

                feat_channels.append(channel_mult)
                in_channel = channel_mult

            if i != n_block - 1:
                down_layers.append(Downsample(in_channel))
                feat_channels.append(in_channel)

        self.down = nn.ModuleList(down_layers)

    def forward(self, input):
        out = spatial_fold(input, self.fold)
        for layer in self.down:
            if isinstance(layer, EncResBlockWithAttention):
                out = layer(out)

            else:
                out = layer(out)
        return out

class EncoderM(BaseEncoder):

    def __init__(
            self,
            in_channel: StrictInt,
            channel: StrictInt,
            channel_multiplier: List[StrictInt],
            n_res_blocks: StrictInt,
            attn_strides: List[StrictInt],
            attn_heads: StrictInt = 1,
            dropout: StrictFloat = 0,
            fold: StrictInt = 1,
    ):
        super().__init__(
            in_channel,
            channel,
            channel_multiplier,
            n_res_blocks,
            attn_strides,
            attn_heads,
            dropout,
            fold)
        group_norm = channel // 4
        in_channel = channel * 4
        self.mid = nn.ModuleList(
            [
                EncResBlockWithAttention(
                    in_channel,
                    in_channel,
                    dropout=dropout,
                    use_attention=True,
                    attention_head=attn_heads,
                    group_norm=group_norm
                ),
                EncResBlockWithAttention(
                    in_channel,
                    in_channel,
                    dropout=dropout,
                    group_norm=group_norm
                ),
            ]
        )

        self.out = nn.Linear(channel * 4 * 8 * 8, 512)

    def forward(self, input):
        x = super().forward(input)
        for layer in self.mid:
            x = layer(x)
        x = self.out(x.flatten(start_dim=1))
        return x


class Autoencoder(nn.Module):

    def __init__(self, encoder, unet, resize=None):
        super().__init__()
        self.encoder = encoder
        self.unet = unet
        self.resize = resize

    def make_latents(self, x0):
        if not self.resize is None:
            x0 = F.interpolate(x0, size=self.resize, mode="bicubic", align_corners=False)
        image_vector = self.encoder(x0)
        return image_vector

    def forward(self, latents, input, time):
        time_embed = self.unet.time(time)
        modulation = torch.cat([time_embed, latents], -1)
        result = self.unet.forward_(input, modulation)
        return result
