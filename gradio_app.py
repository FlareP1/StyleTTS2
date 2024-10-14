import os
import re
import gradio as gr
#import tempfile
#from einops import rearrange
#from vocos import Vocos
#from pydub import AudioSegment
#from model import CFM, UNetT, DiT, MMDiT
#from cached_path import cached_path
#from model.utils import (
#    load_checkpoint,
#    get_tokenizer,
#    convert_char_to_pinyin,
#    save_spectrogram,
#)
#from transformers import pipeline
#import librosa
import click

import torch
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import random
random.seed(0)

import numpy as np
np.random.seed(0)

import time
import random
import yaml
from munch import Munch
import numpy as np
#import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
from nltk.tokenize import word_tokenize

from models import *
from utils import *
from text_utils import TextCleaner
textclenaer = TextCleaner()

to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

def compute_style(path, normalise=False):
    wave, sr = librosa.load(path, sr=24000)
    if (normalise):
        wave = librosa.util.normalize(wave)
    audio, index = librosa.effects.trim(wave, top_db=30)

    if sr != 24000:
        audio = librosa.resample(audio, sr, 24000)
    mel_tensor = preprocess(audio).to(device)

    with torch.no_grad():
        ref_s = model.style_encoder(mel_tensor.unsqueeze(1))
        ref_p = model.predictor_encoder(mel_tensor.unsqueeze(1))

    ref_sp = torch.cat([ref_s, ref_p], dim=1)

    return ref_sp

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

gr.Info("Loading Phonemizer...")

print(f"Using {device} device")

# load phonemizer
import phonemizer
global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True,  with_stress=True)

""" gr.Info("Loading Models...")
config = yaml.safe_load(open("Models/LibriTTS/config.yml"))

# load pretrained ASR model
ASR_config = config.get('ASR_config', False)
ASR_path = config.get('ASR_path', False)
text_aligner = load_ASR_models(ASR_path, ASR_config)

# load pretrained F0 model
F0_path = config.get('F0_path', False)
pitch_extractor = load_F0_models(F0_path)

# load BERT model
from Utils.PLBERT.util import load_plbert
BERT_path = config.get('PLBERT_dir', False)
plbert = load_plbert(BERT_path)

model_params = recursive_munch(config['model_params'])
model = build_model(model_params, text_aligner, pitch_extractor, plbert)
_ = [model[key].eval() for key in model]
_ = [model[key].to(device) for key in model]

params_whole = torch.load("Models/LibriTTS/epochs_2nd_00020.pth", map_location='cpu',weights_only=True)
params = params_whole['net']

for key in model:
    if key in params:
        print('%s loaded' % key)
        try:
            model[key].load_state_dict(params[key])
        except:
            from collections import OrderedDict
            state_dict = params[key]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            # load params
            model[key].load_state_dict(new_state_dict, strict=False)
#             except:
#                 _load(params[key], model[key])
_ = [model[key].eval() for key in model]

from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
sampler = DiffusionSampler(
    model.diffusion.diffusion,
    sampler=ADPM2Sampler(),
    sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
    clamp=False
) """

gr.Info("Loading Punkt...")
import nltk
nltk.download('punkt_tab')

# Setup Global Model variables
model=None
model_params=None
sampler=None
gModelType=None

def normalize_audio(samples):
    # Convert the samples to a numpy array for easier manipulation
    #samples = np.array(samples, dtype=np.float32)

    # Normalize the audio by dividing by the maximum absolute value
    max_amplitude = np.max(np.abs(samples))
    if max_amplitude > 0:
        normalized_samples = samples / max_amplitude
    else:
        normalized_samples = samples  # Avoid division by zero

    return normalized_samples

def loadModel(Config_File, Model_File):
    gr.Info("Loading Models...")

    #Ensure that updates are performed to the global variables
    global model
    global model_params
    global sampler

    #config = yaml.safe_load(open("Models/LibriTTS/config.yml"))
    config = yaml.safe_load(open(Config_File))

    # load pretrained ASR model
    ASR_config = config.get('ASR_config', False)
    ASR_path = config.get('ASR_path', False)
    text_aligner = load_ASR_models(ASR_path, ASR_config)

    # load pretrained F0 model
    F0_path = config.get('F0_path', False)
    pitch_extractor = load_F0_models(F0_path)

    # load BERT model
    from Utils.PLBERT.util import load_plbert
    BERT_path = config.get('PLBERT_dir', False)
    plbert = load_plbert(BERT_path)

    model_params = recursive_munch(config['model_params'])
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    _ = [model[key].eval() for key in model]
    _ = [model[key].to(device) for key in model]

    #params_whole = torch.load("Models/LibriTTS/epochs_2nd_00020.pth", map_location='cpu',weights_only=True)
    params_whole = torch.load(Model_File, map_location='cpu',weights_only=True)
    params = params_whole['net']

    for key in model:
        if key in params:
            print('%s loaded' % key)
            try:
                model[key].load_state_dict(params[key])
            except:
                from collections import OrderedDict
                state_dict = params[key]
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] # remove `module.`
                    new_state_dict[name] = v
                # load params
                model[key].load_state_dict(new_state_dict, strict=False)
    #             except:
    #                 _load(params[key], model[key])
    _ = [model[key].eval() for key in model]

    from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
    sampler = DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
        clamp=False
    )
    return
        
def loadModelType(ModelType):
    global gModelType 
    gModelType = ModelType

    if ModelType=="StyleTTS2-LibriTTS":
        gr.Info("Loading StyleTTS2-LibriTTS Model...")
        loadModel("Models/LibriTTS/config.yml", "Models/LibriTTS/epochs_2nd_00020.pth")
    elif ModelType=="StyleTTS2-LJSpeech":
        gr.Info("Loading StyleTTS2-LJSpeech Model...")
        loadModel("Models/LJSpeech/config.yml", "Models/LJSpeech/epoch_2nd_00100.pth")
    elif ModelType=="APSpeech":
        gr.Info("Loading APSpeech Model...")
        loadModel("Models/APSpeech/config_ft.yml", "Models/APSpeech/epoch_2nd_00049.pth")
    elif ModelType=="ASMRSpeech":
        gr.Info("Loading ASMRSpeech Model...")
        loadModel("Models/ASMRSpeech/config_ft.yml", "Models/ASMRSpeech/epoch_2nd_00049.pth")

    gr.Info("Loading Complete...")
    return      

#model, model_params, sampler = loadModel("Models/LibriTTS/config.yml", "Models/LibriTTS/epochs_2nd_00020.pth")
#model, model_params, sampler = loadModel("Models/LJSpeech/config.yml", "Models/LJSpeech/epoch_2nd_00100.pth")

def inferTTS2(ref_text_input, ref_audio_input=[], alpha = 0.3, beta = 0.7, diffusion_steps=5, embedding_scale=1, speed=1, normalise_output=False, normalise_input=False):
    
    if(gModelType=="StyleTTS2-LJSpeech"):
        noise = torch.randn(1,1,256).to(device)
        gr.Info("Generate Output Audio...")
        wav = inferenceWithOutRef(ref_text_input, noise, diffusion_steps, embedding_scale, speed)
    else:
        gr.Info("Process Input Reference...")
        ref_s = compute_style(ref_audio_input,normalise_input)
        gr.Info("Generate Output Audio...")
        wav = LongFormInference(ref_text_input, ref_s, alpha, beta, diffusion_steps, embedding_scale, speed)

    if(normalise_output):
        gr.Info("Normalising Output Audio...")
        wav = normalize_audio(wav)

    gr.Info("Complete...")
    return (24000, wav)

def inferenceWithRef(text, ref_s, alpha = 0.3, beta = 0.7, diffusion_steps=5, embedding_scale=1, speed=1):
    text = text.strip()
    ps = global_phonemizer.phonemize([text])
    ps = word_tokenize(ps[0])
    ps = ' '.join(ps)
    tokens = textclenaer(ps)
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)
    
    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        text_mask = length_to_mask(input_lengths).to(device)

        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2) 

        s_pred = sampler(noise = torch.randn((1, 256)).unsqueeze(1).to(device), 
                                          embedding=bert_dur,
                                          embedding_scale=embedding_scale,
                                            features=ref_s, # reference from the same speaker as the embedding
                                             num_steps=diffusion_steps).squeeze(1)


        s = s_pred[:, 128:]
        ref = s_pred[:, :128]

        ref = alpha * ref + (1 - alpha)  * ref_s[:, :128]
        s = beta * s + (1 - beta)  * ref_s[:, 128:]

        d = model.predictor.text_encoder(d_en, 
                                         s, input_lengths, text_mask)

        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)

        duration = torch.sigmoid(duration).sum(axis=-1)
        duration = duration*1/speed

        pred_dur = torch.round(duration.squeeze()).clamp(min=1)

        # Eliminate potential noise at the end of the audio during generation.
        if not text[-1].isalnum():
            pred_dur[-1] = 0

        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        # encode prosody
        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(en)
            asr_new[:, :, 0] = en[:, :, 0]
            asr_new[:, :, 1:] = en[:, :, 0:-1]
            en = asr_new

        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

        asr = (t_en @ pred_aln_trg.unsqueeze(0).to(device))
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(asr)
            asr_new[:, :, 0] = asr[:, :, 0]
            asr_new[:, :, 1:] = asr[:, :, 0:-1]
            asr = asr_new

        out = model.decoder(asr, 
                                F0_pred, N_pred, ref.squeeze().unsqueeze(0))
    
        
    return out.squeeze().cpu().numpy()[..., :-50] # weird pulse at the end of the model, need to be fixed later

def LFinferenceWithRef(text, s_prev, ref_s, alpha = 0.3, beta = 0.7, t = 0.7, diffusion_steps=5, embedding_scale=1, speedup=1):
    text = text.strip()
    ps = global_phonemizer.phonemize([text])
    ps = word_tokenize(ps[0])
    ps = ' '.join(ps)
    ps = ps.replace('``', '"')
    ps = ps.replace("''", '"')

    tokens = textclenaer(ps)
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)
    
    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        text_mask = length_to_mask(input_lengths).to(device)

        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2) 
        len=128

        s_pred = sampler(noise = torch.randn((1, len*2)).unsqueeze(1).to(device), 
                                          embedding=bert_dur,
                                          embedding_scale=embedding_scale,
                                            features=ref_s, # reference from the same speaker as the embedding
                                             num_steps=diffusion_steps).squeeze(1)
        
        if s_prev is not None:
            # convex combination of previous and current style
            s_pred = t * s_prev + (1 - t) * s_pred
        
        s = s_pred[:, len:]
        ref = s_pred[:, :len]
        
        ref = alpha * ref + (1 - alpha)  * ref_s[:, :len]
        s = beta * s + (1 - beta)  * ref_s[:, len:]

        s_pred = torch.cat([ref, s], dim=-1)

        d = model.predictor.text_encoder(d_en, 
                                         s, input_lengths, text_mask)

        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x) #speedup by 1/0.95x
       
        duration = torch.sigmoid(duration).sum(axis=-1)
        duration = duration*1/speedup

        pred_dur = torch.round(duration.squeeze()).clamp(min=1)

        # Eliminate potential noise at the end of the audio during generation.
        if not text[-1].isalnum():
          pred_dur[-1] = 0

        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        # encode prosody
        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(en)
            asr_new[:, :, 0] = en[:, :, 0]
            asr_new[:, :, 1:] = en[:, :, 0:-1]
            en = asr_new

        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

        asr = (t_en @ pred_aln_trg.unsqueeze(0).to(device))
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(asr)
            asr_new[:, :, 0] = asr[:, :, 0]
            asr_new[:, :, 1:] = asr[:, :, 0:-1]
            asr = asr_new

        out = model.decoder(asr, 
                                F0_pred, N_pred, ref.squeeze().unsqueeze(0))
    
    return out.squeeze().cpu().numpy()[..., :-8000], s_pred # weird pulse at the end of the model, need to be fixed later
  
def LongFormInference(passage, s_ref, alpha = 0.3, beta = 0.7, diffusion_steps=5, embedding_scale=1, speed=1):
    gr.Info("Generate LongFormInference...")
    sentences = passage.split('.') # simple split by comma
    wavs = []
    s_prev = None
    t = 0.35
    for text in sentences:
        gr.Info("Generate {text}...")
        if text.strip() == "": continue
        #text +=  '.' # add it back
        text = '..... ' + text + ' .....$' # Pad the beginning of the speech with . to prevent the iregular behaviour wiht the first word
    
        wav, s_prev = LFinferenceWithRef(text, 
                            s_prev, 
                            s_ref, 
                            alpha, 
                            beta,  # make it more suitable for the text
                            t, 
                            diffusion_steps,
                            embedding_scale, speed)
        wavs.append(wav)
    gr.Info("Complete...")
    return np.concatenate(wavs)

def inferenceWithOutRef(text, noise, diffusion_steps=5, embedding_scale=1, speed=1):
   
    text = text.strip()
    text = text.replace('"', '')
    ps = global_phonemizer.phonemize([text])
    ps = word_tokenize(ps[0])
    ps = ' '.join(ps)

    tokens = textclenaer(ps)
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)
    
    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(tokens.device)
        text_mask = length_to_mask(input_lengths).to(tokens.device)

        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2) 

        s_pred = sampler(noise, 
              embedding=bert_dur[0].unsqueeze(0), num_steps=diffusion_steps,
              embedding_scale=embedding_scale).squeeze(0)

        s = s_pred[:, 128:]
        ref = s_pred[:, :128]

        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1)
        duration = duration*1/speed

        pred_dur = torch.round(duration.squeeze()).clamp(min=1)

        pred_dur[-1] += 5

        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        # encode prosody
        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
        out = model.decoder((t_en @ pred_aln_trg.unsqueeze(0).to(device)), 
                                F0_pred, N_pred, ref.squeeze().unsqueeze(0))
        
    return out.squeeze().cpu().numpy()

#pipe = pipeline(
#    "automatic-speech-recognition",
#    model="openai/whisper-large-v3-turbo",
#    torch_dtype=torch.float16,
#    device=device,
#)

# --------------------- Settings -------------------- #

#target_sample_rate = 24000
#n_mel_channels = 100
#hop_length = 256
#target_rms = 0.1
#nfe_step = 32  # 16, 32
#cfg_strength = 2.0
#ode_method = "euler"
#sway_sampling_coef = -1.0
#speed = 1.0
# fix_duration = 27  # None or float (duration in seconds)
#fix_duration = None


""" def load_model(exp_name, model_cls, model_cfg, ckpt_step):
    ckpt_path = str(cached_path(f"hf://SWivid/F5-TTS/{exp_name}/model_{ckpt_step}.safetensors"))
    #ckpt_path = f"ckpts/{exp_name}/model_{ckpt_step}.safetensors"  # .pt | .safetensors
    vocab_char_map, vocab_size = get_tokenizer("Emilia_ZH_EN", "pinyin")
    model = CFM(
        transformer=model_cls(
            **model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels
        ),
        mel_spec_kwargs=dict(
            target_sample_rate=target_sample_rate,
            n_mel_channels=n_mel_channels,
            hop_length=hop_length,
        ),
        odeint_kwargs=dict(
            method=ode_method,
        ),
        vocab_char_map=vocab_char_map,
    ).to(device)

    model = load_checkpoint(model, ckpt_path, device, use_ema = True)

    return model """


# load models
#F5TTS_model_cfg = dict(
#    dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4
#)
#E2TTS_model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)

#F5TTS_ema_model = load_model(
#    "F5TTS_Base", DiT, F5TTS_model_cfg, 1200000
#)
#E2TTS_ema_model = load_model(
#    "E2TTS_Base", UNetT, E2TTS_model_cfg, 1200000
#)


""" def infer(ref_audio_orig, ref_text, gen_text, exp_name, remove_silence, speed):
    print(gen_text)
    if len(gen_text) > 40000:
        raise gr.Error("Please keep your text under 40000 chars.")
    gr.Info("Converting audio...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        aseg = AudioSegment.from_file(ref_audio_orig)
        audio_duration = len(aseg)
        if audio_duration > 15000:
            gr.Warning("Audio is over 15s, clipping to only first 15s.")
            aseg = aseg[:15000]
        aseg.export(f.name, format="wav")
        ref_audio = f.name
    if exp_name == "F5-TTS":
        ema_model = F5TTS_ema_model
    elif exp_name == "E2-TTS":
        ema_model = E2TTS_ema_model

    if not ref_text.strip():
        gr.Info("No reference text provided, transcribing reference audio...")
        ref_text = outputs = pipe(
            ref_audio,
            chunk_length_s=30,
            batch_size=128,
            generate_kwargs={"task": "transcribe"},
            return_timestamps=False,
        )["text"].strip()
        gr.Info("Finished transcription")
    else:
        gr.Info("Using custom reference text...")
    audio, sr = torchaudio.load(ref_audio)
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    rms = torch.sqrt(torch.mean(torch.square(audio)))
    if rms < target_rms:
        audio = audio * target_rms / rms
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        audio = resampler(audio)
    audio = audio.to(device)

    # Prepare the text
    text_list = [ref_text + gen_text]
    final_text_list = convert_char_to_pinyin(text_list)

    # Calculate duration
    ref_audio_len = audio.shape[-1] // hop_length
    # if fix_duration is not None:
    #     duration = int(fix_duration * target_sample_rate / hop_length)
    # else:
    zh_pause_punc = r"。，、；：？！"
    ref_text_len = len(ref_text) + len(re.findall(zh_pause_punc, ref_text))
    gen_text_len = len(gen_text) + len(re.findall(zh_pause_punc, gen_text))
    #duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / speed)
    duration = int(ref_text_len / ref_audio_len  * gen_text_len / speed  )
    duration = int( duration * target_sample_rate / hop_length )
    gr.Info(f"Estimated Audio Length {duration}")

    # inference
    gr.Info(f"Generating audio using {exp_name}")
    with torch.inference_mode():
        generated, _ = ema_model.sample(
            cond=audio,
            text=final_text_list,
            duration=duration,
            steps=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
        )

    generated = generated[:, ref_audio_len:, :]
    generated_mel_spec = rearrange(generated, "1 n d -> 1 d n")
    gr.Info("Running vocoder")
    vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")
    generated_wave = vocos.decode(generated_mel_spec.cpu())
    if rms < target_rms:
        generated_wave = generated_wave * rms / target_rms

    # wav -> numpy
    generated_wave = generated_wave.squeeze().cpu().numpy()

    if remove_silence:
        gr.Info("Removing audio silences... This may take a moment")
        non_silent_intervals = librosa.effects.split(generated_wave, top_db=30)
        non_silent_wave = np.array([])
        for interval in non_silent_intervals:
            start, end = interval
            non_silent_wave = np.concatenate(
                [non_silent_wave, generated_wave[start:end]]
            )
        generated_wave = non_silent_wave

    # spectogram
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spectrogram:
        spectrogram_path = tmp_spectrogram.name
        save_spectrogram(generated_mel_spec[0].cpu().numpy(), spectrogram_path)

    return (target_sample_rate, generated_wave), spectrogram_path
 """

with gr.Blocks() as app:
    gr.Markdown(
        """
# StyleTTS 2

This is a local web UI for StyleTTS2, . This app supports the following TTS models:

* [StyleTTS2-LJSpeech](https://huggingface.co/yl4579/StyleTTS2-LJSpeech) 
* [StyleTTS2-LibriTTS](https://huggingface.co/yl4579/StyleTTS2-LibriTTS)

https://arxiv.org/abs/2306.07691
See the original repository for details https://github.com/yl4579/StyleTTS2

If you're having issues, try converting your reference audio to WAV or MP3, clipping it to 15s, and shortening your prompt.

"""
    )

    ref_audio_input = gr.Audio(label="Reference Audio", type="filepath")
    gen_text_input = gr.Textbox(value="This is a TTS text to speech model that uses style diffusion to create human like speech, can you tell that this is not real?", label="Text to Generate (max 200 chars.)", lines=4)
    model_choice = gr.Radio(
        choices=["StyleTTS2-LJSpeech", "StyleTTS2-LibriTTS","APSpeech","ASMRSpeech"], label="Choose TTS Model", value="StyleTTS2-LibriTTS"
    )

    load_btn = gr.Button("Load Model", variant="primary")
    generate_btn = gr.Button("Synthesize", variant="primary")
    with gr.Accordion("Advanced Settings", open=False):
        alpha = gr.Number(value=0.3,label="Alpha",step=0.1)
        beta = gr.Number(value=0.7,label="Beta",step=0.1)
        diffusion_steps=gr.Number(value=10,label="Steps")
        embedding_scale=gr.Number(value=1.5,label="Embedding Scale")
        speed=gr.Number(value=1,label="Speed Up",step=0.05)
        
        normalise_output = gr.Checkbox(
            label="Normalise Output",
            info="The model can produce quite output if your reference is quite.",
            value=False,
            interactive=True,
        )

        normalise_intput = gr.Checkbox(
            label="Normalise Input",
            info="The model can produce quite output if your reference is quite.",
            value=False,
            interactive=True,
        )

    audio_output = gr.Audio(label="Synthesized Audio")
    spectrogram_output = gr.Image(label="Spectrogram")

    load_btn.click(
        loadModelType,
        inputs=[
            model_choice,
        ],
       outputs=[],
    )

    generate_btn.click(
        inferTTS2,
        inputs=[
            gen_text_input,
            ref_audio_input,
            alpha,
            beta,
            diffusion_steps,
            embedding_scale,
            speed,
            normalise_output,
            normalise_intput,
        ],
        outputs=[audio_output],
    )


@click.command()
@click.option("--port", "-p", default=None, type=int, help="Port to run the app on")
@click.option("--host", "-H", default=None, help="Host to run the app on")
@click.option(
    "--share",
    "-s",
    default=False,
    is_flag=True,
    help="Share the app via Gradio share link",
)
@click.option("--api", "-a", default=True, is_flag=True, help="Allow API access")
def main(port, host, share, api):
    global app
    print(f"Starting app...")
    app.queue(api_open=api).launch(
        server_name=host, server_port=port, share=share, show_api=api
    )


if __name__ == "__main__":
    main()
