from Modules.Bert import Bert


class Model_loader(object):
    def __init__(self, config, data_generator, device, vocab_num=0):
        self.model = Bert(bertName=config['bertName'], has_neighbor=config['has_neighbor'],
                              dropout=config['dropout'], vocab_num=vocab_num, device=device,
                              has_mlm=config['has_mlm'],
                              node_num=data_generator.node_num, pretrainEmb=data_generator.embedding,
                              pred_type=config["pred_type"], graph=data_generator.graph,
                              mskrate=0.2).to(device)
