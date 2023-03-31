from argparse import ArgumentParser
from ssl import OP_NO_TLSv1_1

BATCHNORM_MOMENTUM = 0.01

class Config(object):
    """Wrapper class for model hyperparameters."""

    def __init__(self):
        """
        Defaults
        """
        self.mode = None
        self.save_path = None
        self.model_path = None
        self.data_path = None
        self.datasize = None
        self.ckpt = None
        self.optimizer = None
        self.bce_loss = None
        self.lr = 1e-4     ##change from 1e-5 to 1e-4
        self.enc_layer = 1  ## set to 1
        self.dec_layer = 1
        self.nepoch = 10
        self.lr_backbone=1e-5
        self.weight_decay=1e-4
        self.epochs=300

        
        self.parser = self.setup_parser()
        self.args = vars(self.parser.parse_args())
        self.__dict__.update(self.args)

    def setup_parser(self):
        """
        Sets up an argument parser
        :return:
        """
        parser = ArgumentParser(description='training code')
        # parser.add_argument('config', help='test config file path')
        parser.add_argument('--lr', default=1e-4, type=float)
        parser.add_argument('--lr_backbone', default=1e-5, type=float)
        parser.add_argument('--batch_size', default=2, type=int)
        parser.add_argument('--weight_decay', default=1e-4, type=float)
        parser.add_argument('--epochs', default=300, type=int)
        parser.add_argument('--lr_drop', default=200, type=int)
        parser.add_argument('--clip_max_norm', default=0.1, type=float,
                            help='gradient clipping max norm')

        # Model parameters
        parser.add_argument('--frozen_weights', type=str, default=None,
                            help="Path to the pretrained model. If set, only the mask head will be trained")
        # * Backbone
        ###改成了resnet18，原本为resnet50
        parser.add_argument('--backbone', default='resnet50', type=str,
                            help="Name of the convolutional backbone to use")
        parser.add_argument('--dilation', action='store_true',
                            help="If true, we replace stride with dilation in the last convolutional block (DC5)")
        parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                            help="Type of positional embedding to use on top of the image features")

        # * Transformer
        parser.add_argument('--enc_layers', default=6, type=int,
                            help="Number of encoding layers in the transformer")
        parser.add_argument('--dec_layers', default=6, type=int,
                            help="Number of decoding layers in the transformer")
        parser.add_argument('--dim_feedforward', default=2048, type=int,
                            help="Intermediate size of the feedforward layers in the transformer blocks")
        parser.add_argument('--hidden_dim', default=256, type=int,
                            help="Size of the embeddings (dimension of the transformer)")
        parser.add_argument('--dropout', default=0.1, type=float,
                            help="Dropout applied in the transformer")
        parser.add_argument('--nheads', default=8, type=int,
                            help="Number of attention heads inside the transformer's attentions")
        parser.add_argument('--num_queries', default=100, type=int,
                            help="Number of query slots")
        parser.add_argument('--pre_norm', action='store_true')

        # * Segmentation
        parser.add_argument('--masks', action='store_true',
                            help="Train segmentation head if the flag is provided")

        # Loss
        parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                            help="Disables auxiliary decoding losses (loss at each layer)")
        # * Matcher
        parser.add_argument('--set_cost_class', default=1, type=float,
                            help="Class coefficient in the matching cost")
        parser.add_argument('--set_cost_bbox', default=5, type=float,
                            help="L1 box coefficient in the matching cost")
        parser.add_argument('--set_cost_giou', default=2, type=float,
                            help="giou box coefficient in the matching cost")
        # * Loss coefficients
        parser.add_argument('--mask_loss_coef', default=1, type=float)
        parser.add_argument('--dice_loss_coef', default=1, type=float)
        parser.add_argument('--bbox_loss_coef', default=5, type=float)
        parser.add_argument('--giou_loss_coef', default=2, type=float)
        parser.add_argument('--eos_coef', default=0.1, type=float,
                            help="Relative classification weight of the no-object class")

        # dataset parameters
        parser.add_argument('--dataset_file', default='coco')
        parser.add_argument('--coco_path', type=str)
        parser.add_argument('--coco_panoptic_path', type=str)
        parser.add_argument('--remove_difficult', action='store_true')

        parser.add_argument('--output_dir', default='',
                            help='path where to save, empty for no saving')
        parser.add_argument('--device', default='cuda',
                            help='device to use for training / testing')
        parser.add_argument('--seed', default=42, type=int)
        parser.add_argument('--resume', default='', help='resume from checkpoint')
        parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                            help='start epoch')
        parser.add_argument('--eval', action='store_true')
        parser.add_argument('--num_workers', default=2, type=int)

        # distributed training parameters
        parser.add_argument('--world_size', default=1, type=int,
                            help='number of distributed processes')
        parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
        ###STTran
        parser.add_argument('-mode', dest='mode', help='predcls/sgcls/sgdet', default='predcls', type=str)
        parser.add_argument('-save_path', default='data1/', type=str)
        parser.add_argument('-model_path', default=None, type=str)
        parser.add_argument('-data_path', default='/data/scene_understanding/action_genome/', type=str)
        parser.add_argument('-datasize', dest='datasize', help='mini dataset or whole', default='large', type=str)
        parser.add_argument('-ckpt', dest='ckpt', help='checkpoint', default=None, type=str)
        parser.add_argument('-optimizer', help='adamw/adam/sgd', default='adamw', type=str)
        parser.add_argument('-lr', dest='lr', help='learning rate', default=1e-5, type=float)
        parser.add_argument('-nepoch', help='epoch number', default=10, type=float)
        parser.add_argument('-bce_loss', action='store_true')
        parser.add_argument('-checkpoint_path', type=str,default='./checkpoint')

        #
        parser.add_argument('-use_wandb', default=None,type=str)
        return parser

