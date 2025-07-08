from __future__ import print_function
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import logging
import torch
import torch.nn as nn
from data.data_recall import data_generator
from utils.lib import dump_json, set_seed, count_para
from torch.optim.lr_scheduler import StepLR
from model.spiking_seqmnist_model import TESNN
from model.Hyperparameters_recall import args
from model.neuron import ALIF, RhythmALIF, DEXAT, RhythmDEXAT
from model.lstm import lstm

current_dir = os.path.dirname(os.getcwd())



set_seed(args.seed)

if __name__ == "__main__":


    home_dir = current_dir  # relative path
    snn_ckp_dir = os.path.join(home_dir, 'Delay_recall_task/exp/Delayed_recall/checkpoint/')
    snn_rec_dir = os.path.join(home_dir, 'Delay_recall_task/exp/Delayed_recall/record/')

    seq_length = args.seq_len
    # logging.info(f"Sequence length: {seq_length}")
    delay_length = 10
    max_duration = args.max_duration
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr

    X_train, Y_train = data_generator(delay_length, seq_length, 50000, encode=True)
    X_test, Y_test = data_generator(delay_length, seq_length, 1000, encode=True)


    num_epochs = args.epochs
    learning_rate = args.lr
    arch_name = '-'.join([str(s) for s in args.fc])



    if args.te == 'LSTM':
        train_record_path = f"LSTM_seq{args.seq_len}_delay{delay_length}_co{args.beta}_lens{args.lens}_arch{arch_name}_lr{learning_rate}"
        net = lstm(INPUT_SIZE=9, OUT_SIZE=10, LAYERS=len(args.fc), HIDDEN_SIZE=args.fc[0])
    elif args.te == 'ALIF':
        train_record_path = f"ALIF_seq{args.seq_len}_delay{delay_length}_co{args.beta}_lens{args.lens}_arch{arch_name}_lr{learning_rate}"
        net = ALIF(time_window=delay_length + 2 * seq_length, input_size=5, output_size=5, hidden_dims=args.fc)
    elif args.te == 'DEXAT':
        train_record_path = f"DEXAT_seq{args.seq_len}_delay{delay_length}_co{args.beta}_lens{args.lens}_arch{arch_name}_lr{learning_rate}"
        net = DEXAT(time_window=delay_length + 2 * seq_length, input_size=5, output_size=5, hidden_dims=args.fc)
    elif args.te == 'RhythmALIF':
        train_record_path = f"RhythmALIF_seq{args.seq_len}_delay{delay_length}_co{args.beta}_lens{args.lens}_arch{arch_name}_lr{learning_rate}"
        net = RhythmALIF(time_window=delay_length + 2 * seq_length, input_size=5, output_size=5, hidden_dims=args.fc)
    elif args.te == 'RhythmDEXAT':
        train_record_path = f"RhythmDEXAT_seq{args.seq_len}_delay{delay_length}_co{args.beta}_lens{args.lens}_arch{arch_name}_lr{learning_rate}"
        net = RhythmDEXAT(time_window=delay_length + 2 * seq_length, input_size=5, output_size=5, hidden_dims=args.fc)
    else:
        raise ValueError(f"Unknown spiking model: {args.te}")

    train_record_path = train_record_path + f"_seed_{args.seed}"
    train_chk_pnt_path = train_record_path

    # Set up logging

    # logging_path = os.path.join(home_dir, 'Delay_recall_task/exp/Delayed_recall/loggingrecord/')
    # train_full_record_path = os.path.join(logging_path, train_record_path + ".log")
    # logging.basicConfig(
    #     filename=train_full_record_path,
    #     filemode="a",
    #     level=logging.INFO,
    #     format="%(asctime)s - %(levelname)s - %(message)s"
    # )

    logging_path = os.path.join(home_dir, 'Delay_recall_task/exp/Delayed_recall/loggingrecord/')
    train_full_record_path = os.path.join(logging_path, train_record_path + ".log")

    # Ensure the logging directory exists
    if not os.path.exists(logging_path):
        os.makedirs(logging_path)

    # Create a logger
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.DEBUG)  # Set the logger level to DEBUG (lowest level to capture everything)

    # Create a file handler
    file_handler = logging.FileHandler(train_full_record_path)
    # file_handler.setLevel(logging.DEBUG)  # Log all levels to the file
    file_handler.setLevel(logging.INFO)  # Log INFO levels to the file

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Log INFO and above to the terminal

    # Create a formatter and set it for both handlers
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # CUDA configuration
    if torch.cuda.is_available():
        device = 'cuda'
        logger.info("GPU is available.")
    else:
        device = 'cpu'
        logger.info("GPU is not available.")

    # Log details about the architecture
    logger.info(
        f"Sequence length: {args.seq_len}, Delay length: {delay_length}, Algorithm: {args.te}, "
        f"Threshold: {args.thresh}, Lens: {args.lens}, Decay: {args.decay}, Input size: {args.in_size}, "
        f"Learning rate: {args.lr}, Beta: {args.beta}"
    )
    logger.info(f"Architecture: {arch_name}")



    net = net.to(device)
    logger.info(f"Network architecture: \n{net}")
    para = count_para(net)
    logger.info(f"Parameter count: {para}")

    X_train = X_train.to(device)
    Y_train = Y_train.to(device)
    X_test = X_test.to(device)
    Y_test = Y_test.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.8)
    criterion = nn.MSELoss()

    best_loss = 1000
    loss_train_record = []
    loss_test_record = []
    acc_train_record = []
    acc_test_record = []

    def train(epoch):
        correct = 0
        counter = 0
        net.train()
        batch_idx = 1
        total_loss = 0
        for i in range(0, X_train.size(0), batch_size):
            if i + batch_size > X_train.size(0):
                x, y = X_train[i:], Y_train[i:]
            else:
                x, y = X_train[i:(i + batch_size)], Y_train[i:(i + batch_size)]
            optimizer.zero_grad()
            output = net(x, 'recall')
            output = output.transpose(1, 2)
            loss = criterion(output.reshape(-1, 5), y.transpose(1, 2).reshape(-1, 5))
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            optimizer.step()
            batch_idx += 1
            total_loss += loss.item()

            pred = output[:, -seq_length:, :].reshape(-1, 5).data.round()
            y = y.transpose(1, 2)
            correct_com = pred.eq(y[:, -seq_length:, :].reshape(-1, 5).data)
            correct_all = correct_com.all(dim=-1)
            correct += correct_all.long().cpu().sum().item()
            counter += pred.size(0)

            current_lr = optimizer.param_groups[0]['lr']
            if batch_idx % 40 == 0:
                avg_loss = total_loss / 40
                logger.info(
                    f"| Epoch {epoch:3d} | {batch_idx:5d}/{50000 // batch_size + 1:5d} batches | lr {current_lr:.5f} | "
                    f"loss {avg_loss:.8f} | accuracy {100. * correct / counter:.4f}"
                )
                loss_train_record.append(avg_loss)
                acc_train_record.append(100. * correct / counter)
                total_loss = 0
                correct = 0
                counter = 0

    def evaluate(epoch):
        global best_loss
        net.eval()
        with torch.no_grad():
            output = net(X_test, 'recall')
            test_loss = criterion(output.transpose(1, 2).reshape(-1, 5), Y_test.transpose(1, 2).reshape(-1, 5))

            pred = output[..., -seq_length:].transpose(1, 2).reshape(-1, 5).data.round()
            correct_com = pred.eq(Y_test[..., -seq_length:].transpose(1, 2).reshape(-1, 5).data)
            correct_all = correct_com.all(dim=-1)
            acc = correct_all.long().cpu().sum().item() / (output.size(0) * seq_length)

            logger.info(f"\nTest set: Average loss: {test_loss:.6f}, Accuracy: {100. * acc:.4f}\n")

            if test_loss < best_loss:
                if not os.path.isdir(snn_ckp_dir):
                    os.makedirs(snn_ckp_dir)
                best_loss = test_loss
                state = {
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': test_loss,
                }
                torch.save(state, snn_ckp_dir + train_chk_pnt_path)
                logger.info("Saving model.")
            loss_test_record.append(test_loss)
            acc_test_record.append(100. * acc)
            return test_loss.item()

    for ep in range(1, epochs + 1):
        train(ep)
        loss = evaluate(ep)
        scheduler.step()
        if not os.path.isdir(snn_ckp_dir):
            os.makedirs(snn_ckp_dir)

        training_record = {
            'learning_rate': args.lr,
            'algo': args.algo,
            'thresh': args.thresh,
            'lens': args.lens,
            'decay': args.decay,
            'architecture': args.fc,
            'loss_test_record': [x.item() if isinstance(x, torch.Tensor) else x for x in loss_test_record],
            'loss_train_record': [x.item() if isinstance(x, torch.Tensor) else x for x in loss_train_record],
            'acc_train_record': [x.item() if isinstance(x, torch.Tensor) else x for x in acc_train_record],
            'acc_test_record': [x.item() if isinstance(x, torch.Tensor) else x for x in acc_test_record],
        }

        dump_json(training_record, snn_rec_dir, train_record_path)