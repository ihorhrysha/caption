from models.resnet_152_lstm import ResNet152LSTM
from models.resnet_50_lstm import ResNet50LSTM

import torch
from models.init_net import init_net


class ModelFactory(object):
    """
    Model simple factory method
    """

    @staticmethod
    def create(params, vocab_size):
        """
        Creates Model based on detector type
        :param params: Model settings
        :return: Model instance. In case of unknown Model type throws exception.
        """

        if params['MODEL']['name'] == 'resnet_152_lstm':
            model = ResNet152LSTM(
                embed_size=params['MODEL']['embed_size'],
                hidden_size=params['MODEL']['hidden_size'],
                vocab_size=vocab_size,
                num_layers=params['MODEL']['num_layers']
            )
        elif params['MODEL']['name'] == 'resnet_50_lstm':
            model = ResNet50LSTM(
                embed_size=params['MODEL']['embed_size'],
                hidden_size=params['MODEL']['hidden_size'],
                vocab_size=vocab_size,
                num_layers=params['MODEL']['num_layers']
            )
        else:
            raise ValueError("ModelFactory(): Unknown Model type: " + params['Model']['type'])

        # if len(params['MODEL']['weights']) > 0:
        #     net.load_state_dict(torch.load(params['MODEL']['weights']))
        # else:
        #     init_net(net, params['MODEL']['init'])

        model = model.to(params['device'])

        return model
