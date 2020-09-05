python3 inference.py -a ../adaptive_voice_conversion/vctk/trimmed_vctk_spectrograms/sr_24000_mel_norm/attr.pkl -c config.yaml -m model/adaptive_vc/G_999.ckpt -s $1 -t $2 -o $3
