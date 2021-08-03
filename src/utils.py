import torchaudio

def torch_sox_effect_load(mp3_path, sr):
    effects = [
        ['rate', str(sr)]
    ]
    waveform, source_sr = torchaudio.load(mp3_path)
    if source_sr != sr:
        waveform, _ = torchaudio.sox_effects.apply_effects_tensor(waveform, source_sr, effects, channels_first=True)
    return waveform