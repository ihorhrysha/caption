from models.resnet_152_lstm import DecoderRNN, EncoderCNN

import torch
from models.init_net import init_net


class ModelFactory(object):
    """
    Model simple factory method
    """

    @staticmethod
    def create(params, vocab):
        """
        Creates Model based on detector type
        :param params: Model settings
        :return: Model instance. In case of unknown Model type throws exception.
        """

        if params['MODEL']['name'] == 'resnet_152_lstm':
            
            encoder = EncoderCNN(
                embed_size=params['MODEL']['encoder']['embed_size']
            )
            decoder = DecoderRNN(
                embed_size=params['MODEL']['encoder']['embed_size'],
                hidden_size=params['MODEL']['decoder']['hidden_size'],
                vocab_size=len(vocab),
                num_layers=params['MODEL']['decoder']['num_layers']
            )
        else:
            raise ValueError("ModelFactory(): Unknown Model type: " + params['Model']['type'])

        # if len(params['MODEL']['weights']) > 0:
        #     net.load_state_dict(torch.load(params['MODEL']['weights']))
        # else:
        #     init_net(net, params['MODEL']['init'])

        encoder = encoder.to(params['device'])
        decoder = decoder.to(params['device'])

        return encoder, decoder
