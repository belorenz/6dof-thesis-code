#!/usr/bin/env python2

import csv
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def print_losses_epochs():
    file_loss_train = '/home/benjamin/Documents/Studium/Informatik/abschlussarbeit/dataset/sample/loss_train_complete.csv'
    file_loss_valid = '/home/benjamin/Documents/Studium/Informatik/abschlussarbeit/dataset/sample/loss_test_complete.csv'

    num_of_batches = 1057
    num_of_batches_test = 118
    resolution = 20.0
    loss_train = print_epochs(file_loss_train, num_of_batches)
    loss_val = print_epochs(file_loss_valid, num_of_batches_test)

    relation = float(len(loss_train)) / float(len(loss_val))
    plt.plot(  loss_train, 'b', label='Training Loss')
    #plt.plot(  loss_val, 'r', label='Validation loss')
    plt.plot( range(0,len(loss_val*int(relation)), int(relation)), loss_val,
              'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.show()

def print_losses_batches():
    file_loss_train = '/home/benjamin/Documents/Studium/Informatik/abschlussarbeit/dataset/sample/loss_train_complete.csv'
    file_loss_valid = '/home/benjamin/Documents/Studium/Informatik/abschlussarbeit/dataset/sample/loss_test_complete.csv'

    loss_train = print_batches(file_loss_train)
    loss_val = print_batches(file_loss_valid)
    relation = float(len(loss_train)) / float(len(loss_val))

    plt.plot( range(len(loss_train)), loss_train, 'b', label='Training loss')
    plt.plot( range(0,len(loss_val*int(relation)), int(relation)), loss_val,
                    'r', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Batches')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.show()


#def print_epochs_deprecated(filename):
#    loss_values = []
#    last_batch_first_file = 0
#    with open(filename, 'rb') as csvfile:
#        loss_reader = csv.reader(csvfile)
#        epoch = 1
#        batch_id = accumulated_loss = 0
#        last_epoch_first_csv_file = 0
#        for i, row in enumerate(loss_reader):
#            if i is 0:
#                continue
#
#            new_epoch = conv(row[0], int) + last_epoch_first_csv_file
#            if new_epoch > epoch:
#                epoch = new_epoch
#                loss_values.append(accumulated_loss / batch_id)
#                accumulated_loss = 0
#            elif new_epoch < epoch and i is not 1:
#                last_epoch_first_csv_file = epoch
#                epoch = new_epoch
#                loss_values.append(accumulated_loss / batch_id)
#                accumulated_loss = 0
#
#            batch_id = conv(row[1], int)
#            accumulated_loss += conv(row[2], float)
#    return loss_values

def print_epochs(filename, intervall):
    loss_values = []
    conti = False
    with open(filename, 'rb') as csvfile:
        loss_reader = csv.reader(csvfile)
        accumulated_loss = 0
        for i, row in enumerate(loss_reader):
            if i is 0:
                continue

            if i % intervall == 0:
                if accumulated_loss == 0:
                    accumulated_loss += conv(row[2], float)
                    conti = True
                loss_values.append(accumulated_loss / intervall)
                accumulated_loss = 0
                if conti:
                    continue
            accumulated_loss += conv(row[2], float)
    return loss_values

def print_batches(filename):
    loss_values = []
    with open(filename, 'rb') as csvfile:
        loss_reader = csv.reader(csvfile)
        for i, row in enumerate(loss_reader):
            if i is 0:
                continue
            loss_values.append(conv(row[2], float))
    return loss_values

def conv(s, type):
    try:
        s=type(s)
    except ValueError:
        pass
    return s



if __name__ == '__main__':
    print_losses_batches()
    print_losses_epochs()
    print('\ndone.')
