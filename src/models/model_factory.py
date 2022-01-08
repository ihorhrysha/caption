from models.resnet_lstm import ResNet152LSTM, ResNet50LSTM, ResNe101LSTM, ResNetLSTM


class ModelFactory(object):
    """
    Model simple factory method
    """

    @staticmethod
    def create(params, vocab_size) -> ResNetLSTM:
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
        elif params['MODEL']['name'] == 'resnet_101_lstm':
            model = ResNe101LSTM(
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

        model = model.to(params['device'])

        return model
