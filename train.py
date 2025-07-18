from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

from datasets import load_dataset
from tokenizers import Tokenizer, normalizers
from tokenizers import models
from tokenizers.trainers import WordLevelTrainer, WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path


import warnings

from config import get_weights_file_path, get_config
from dataset import BilingualDataset, casual_mask
from model import build_transformer



def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    if Path.exists(tokenizer_path):
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    else:
        tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
        tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
        tokenizer.pre_tokenizers = Whitespace()
        # trainer = WordLevelTrainer(special_tokens = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        trainer = WordPieceTrainer(special_tokens = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)

        # tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.train_from_iterator(get_all_content(ds, lang), trainer=trainer)
        # encoding = tokenizer.encode("This is one sentence.", "With this one we have a pair.")

    tokenizer.save(str(tokenizer_path))
    return tokenizer


def get_all_content(ds, lang):
    all_items = [item[lang] for item in ds]
    return all_items


def get_all_sentences(ds, lang):
    print(f"lang: {lang}")
    for item in ds:
        yield item[lang]


def get_ds(config):
    src_lang = config['lang_src']
    tgt_lang = config['lang_tgt']
    # ds_raw = load_dataset('intelsense/bengali-translation-data', f"de_{src_lang}_{tgt_lang}", split="train")
    ds_raw = load_dataset('intelsense/bengali-translation-data', split=f"de_{src_lang}_{tgt_lang}")


   # build tokenizers
    # tokenizer_src = get_or_build_tokenizer(config, ds_raw, src_lang)
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, "lang_2")

    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, tgt_lang)

    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, config['seq_len'])


    max_len_src = 0
    max_len_tgt = 0
    for item in ds_raw:
        # src_ids = tokenizer_src.encode(item['translation'][src_lang]).ids
        src_ids = tokenizer_src.encode(item["lang_2"]).ids
        tgt_ids = tokenizer_tgt.encode(item[tgt_lang]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max length of source sentence - {max_len_src}")
    print(f"Max length of target sentence - {max_len_tgt}")

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model


def train_model(config):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print(f"Using device - {device}")
    device = torch.device(device)
    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    #Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0

    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state["global_step"]

    loss_fn = nn.CrossEntropyLoss(ignore_index = tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)


    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm()
        for batch in tqdm(train_dataloader, desc=f"Processing epoch {epoch: 02d}"):
            # model.train()
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)

            label = batch["label"].to(device)

            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            writer.add_scalar("train loss", loss.item(), global_step)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

            global_step += 1

        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }, model_filename)

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_src.token_to_id('[SOS]')
    eos_idx = tokenizer_src.token_to_id('[EOS]')
    # Compute the encoder input and reuse it for each token generation from decoder
    encoder_output = model.encode(source, source_mask)
    # decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(source).to(device)
    decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(source).to(torch.int64).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break
        decoder_mask = casual_mask(decoder_input.size(1)).type_as(source).to(device)
        output = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        prob = model.project(output[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat([decoder_input, torch.empty(1, 1).fill_(next_word.item()).to(device)], dim=1)
        if next_word.item() == eos_idx:
            break
    return decoder_input.squeeze(0)


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_state, writer,
                   num_examples=2):
    count = 0
    model.eval()
    source_texts = []
    expected = []
    predicted = []
    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)

            assert encoder_input.size(0) == 1, "Validation size must be 1"
            model_output = greedy_decode(model,  encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)
            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_text_output = tokenizer_tgt.decode(model_output.to(torch.int64).detach().cpu().numpy())
            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_text_output)

            print_msg(f"SOURCE TEXT - {source_text}")
            print_msg(f"EXPECTED TEXT - {target_text}")
            print_msg(f"PREDICTED TEXT - {predicted}")

            if count == num_examples:
                    break


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)














