import torch
import torch.nn as nn
from torch import optim

from network import *
from gen_training_loader import DataLoader, collate_fn, SpeechData
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

from multiprocessing import cpu_count
import numpy as np
import argparse
import os
import time


def get_embedding_input(em_infos, length):
    # print(length)
    output = list()
    for em_info in em_infos:
        embeddings = em_info[0]
        sep_list = em_info[1]
        one_batch = list()
        for j in range(len(sep_list)-1):
            temp = torch.stack([embeddings[j]
                                for i in range(sep_list[j+1]-sep_list[j])])
            one_batch.append(temp)
        zero_pad = length - sep_list[len(sep_list)-1]
        # print(zero_pad)
        if zero_pad > 0:
            # print(zero_pad)
            zeros = torch.stack([torch.zeros(768) for i in range(zero_pad)])
            # print(zeros.size())
            one_batch.append(zeros)
        cat_temp = torch.cat(one_batch, 0)
        cat_temp = cat_temp[0:length, :]
        if cat_temp.size(0) < length:
            # print("###############")
            cat_temp = torch.cat([cat_temp, torch.stack(
                [torch.zeros(768) for i in range(length-cat_temp.size(0))])], 0)
            # print(cat_temp.size())
            # cat_temp = 0
        # print(cat_temp.size())
        cat_temp = cat_temp[0:length, :]
        output.append(cat_temp)
    output = torch.stack(output)
    # print(output.size())
    # output = output[:, 0:length, :]

    # print(output.size())
    return output


def main(args):
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define model
    model = nn.DataParallel(Tacotron()).to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model_bert = BertModel.from_pretrained('bert-base-uncased')
    print("Models Have Been Defined")

    # Get dataset
    dataset = SpeechData(args.dataset_path, tokenizer, model_bert)
    # print(type(args.dataset_path))
    # print(len(dataset))

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=hp.lr)

    # # Loss for frequency of human register
    # n_priority_freq = int(3000 / (hp.sample_rate * 0.5) * hp.num_freq)

    # Get training loader
    print("Get Training Loader")
    training_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                 collate_fn=collate_fn, drop_last=True, num_workers=cpu_count())
    # print(len(training_loader))

    # Load checkpoint if exists
    try:
        checkpoint = torch.load(os.path.join(
            hp.checkpoint_path, 'checkpoint_%d.pth.tar' % args.restore_step))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("---Model Restored at Step %d---\n" % args.restore_step)

    except:
        print("---Start New Training---\n")
        if not os.path.exists(hp.checkpoint_path):
            os.mkdir(hp.checkpoint_path)

    # Training
    model = model.train()

    total_step = hp.epochs * len(training_loader)
    # print(total_step)
    # Loss = []
    Time = np.array([])
    Start = time.clock()
    for epoch in range(hp.epochs):
        # print("########")
        for i,  data_batch in enumerate(training_loader):
            start_time = time.clock()
            # print("in")
            # Count step
            current_step = i + args.restore_step + \
                epoch * len(training_loader) + 1
            # print(current_step)

            # Init
            optimizer.zero_grad()

            #  {"text": texts, "mel": mels, "spec": specs}
            texts = data_batch["text"]
            em_infos = data_batch["em_infos"]
            # print(len(em_info))
            # print(texts)
            mels = trans(data_batch["mel"])
            # mels = trans(mels)
            # print(np.shape(mels))
            specs = trans(data_batch["spec"])
            # print(np.shape(specs))
            # mel_input = mels[:, :-1, :]
            # print(np.shape(mel_input))
            # mel_input = mel_input[:, :, -hp.num_mels:]
            # print(np.shape(mel_input))
            # print(np.shape(mels))
            frame_arr = np.zeros(
                [args.batch_size, hp.num_mels, 1], dtype=np.float32)
            # print(np.shape(frame_arr))
            # print(np.shape(mels[:, :, 1:]))
            mel_input = np.concatenate((frame_arr, mels[:, :, 1:]), axis=2)
            # print(np.shape(mel_input))
            # print(mels)

            if torch.cuda.is_available():
                texts = torch.from_numpy(texts).type(
                    torch.cuda.LongTensor).to(device)
            else:
                texts = torch.from_numpy(texts).type(
                    torch.LongTensor).to(device)
            # print(texts.size())
            embeddings = get_embedding_input(em_infos, texts.size(1))
            embeddings = embeddings.to(device)
            mels = torch.from_numpy(mels).to(device)
            specs = torch.from_numpy(specs).to(device)
            mel_input = torch.from_numpy(mel_input).to(device)

            # Forward
            mel_output, linear_output = model.forward(
                texts, mel_input, embeddings)
            # print("#####################")
            # print(np.shape(mel_output))
            # print(np.shape(linear_output))
            # print()
            # print(np.shape(mels[:, :, 1:]))
            # print(np.shape(np.transpose(mels.cpu().numpy())))
            # print(np.shape(specs))
            # print(np.shape(np.transpose(mels)))

            # Calculate loss
            # st = time.clock()
            mel_loss = torch.abs(
                mel_output - compare(mel_output, mels[:, :, 1:], device))
            mel_loss = torch.mean(mel_loss)
            linear_loss = torch.abs(
                linear_output - compare(linear_output, specs, device))
            linear_loss = torch.mean(linear_loss)
            loss = mel_loss + hp.loss_weight * linear_loss
            loss = loss.to(device)
            # Loss.append(loss)
            # et = time.clock()
            # print(et - st)
            # print(loss)

            # Backward
            loss.backward()

            # Clipping gradients to avoid gradient explosion
            nn.utils.clip_grad_norm_(model.parameters(), 1.)

            # Update weights
            optimizer.step()

            if current_step % hp.log_step == 0:
                Now = time.clock()
                # print("time per step: %.2f sec" % time_per_step)
                # print("At timestep %d" % current_step)
                # print("linear loss: %.4f" % linear_loss.data[0])
                # print("mel loss: %.4f" % mel_loss.data[0])
                # print("total loss: %.4f" % loss.data[0])
                # print("Epoch [{}/{}], Step [{}/{}], Linear Loss: {:.4f}, Mel Loss: {:.4f}, Total Loss: {:.4f}.".format(
                #     epoch+1, hp.epochs, current_step, total_step, linear_loss.item(), mel_loss.item(), loss.item()))
                str1 = "Epoch [{}/{}], Step [{}/{}], Linear Loss: {:.4f}, Mel Loss: {:.4f}, Total Loss: {:.4f}.".format(
                    epoch+1, hp.epochs, current_step, total_step, linear_loss.item(), mel_loss.item(), loss.item())
                str2 = "Time Used: {:.3f}s, Estimated Time Remaining: {:.3f}s.".format(
                    (Now-Start), (total_step-current_step)*np.mean(Time))
                # print("Time Used: {:.3f}s, Estimated Time Remaining: {:.3f}s.".format(
                # (Now-Start), (total_step-current_step)*np.mean(Time)))
                print(str1)
                print(str2)

                with open("logger.txt", "a")as f_logger:
                    f_logger.write(str1 + "\n")
                    f_logger.write(str2 + "\n")
                    f_logger.write("\n")

            # print(current_step)
            if current_step % hp.save_step == 0:
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(
                )}, os.path.join(hp.checkpoint_path, 'checkpoint_%d.pth.tar' % current_step))
                print("save model at step %d ..." % current_step)

            if current_step in hp.decay_step:
                optimizer = adjust_learning_rate(optimizer, current_step)

            end_time = time.clock()
            Time = np.append(Time, end_time - start_time)
            if len(Time) == hp.clear_Time:
                temp_value = np.mean(Time)
                Time = np.delete(
                    Time, [i for i in range(len(Time))], axis=None)
                Time = np.append(Time, temp_value)
                # print(Time)


def trans(arr):
    return np.stack([np.transpose(ele) for ele in arr])
    # for i, b in enumerate(arr):
    # arr[i] = np.transpose(b)


def adjust_learning_rate(optimizer, step):
    if step == 500000:
        # if step == 20:
        # print("update")
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0005

    elif step == 1000000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0003

    elif step == 2000000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001

    return optimizer


def compare(out, stan, device):
    # for batch_index in range(len(out)):
    #     for i in range(min([np.shape(out)[2], np.shape(stan)[2]])):
    #         torch.abs(out[batch_index][i], stan[batch_index][i])
    # cnt = min([np.shape(out)[2], np.shape(stan)[2]])
    if np.shape(stan)[2] >= np.shape(out)[2]:
        return stan[:, :, :np.shape(out)[2]]
    # return out[:,:,:cnt], stan[:,:,:cnt]
    else:
        frame_arr = np.zeros([np.shape(out)[0], np.shape(out)[1], np.shape(out)[
                             2]-np.shape(stan)[2]], dtype=np.float32)
        return torch.Tensor(np.concatenate((stan.cpu(), frame_arr), axis=2)).to(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str,
                        help='dataset path', default='dataset')
    parser.add_argument('--restore_step', type=int,
                        help='Global step to restore checkpoint', default=0)
    parser.add_argument('--batch_size', type=int,
                        help='Batch size', default=hp.batch_size)
    args = parser.parse_args()
    main(args)
