import argparse

class switch(object):
    value = None
    def __new__(class_, value):
        class_.value = value
        return True

def case(*args):
    return any((arg == switch.value for arg in args))

class NetTester():

    def __init__(self):
        """Consider this as a temporary specific network tester.
        I will make this an abstract class, so that one can add their torchsummary options here.
        Parameters:
            variables (if needed)
            options     --      Switch case scenario (to choose what network to test)
        """
        options = ['encoder',
           'segmentor',
           'generator',
           'decoder',
           'discriminatorT',
           'discriminatorS',
           'discriminatorP'
           ]

        self.dropout_rate = 0.75
        self.init_type = 'normal'
        self.init_gain = 0.01
        self.gpu_ids = ''

        self.num_classes=5

        print ("Welcome to the Network Tester :")
        print (options)

    def parse(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--network', required=True, help='choose from the networks above. We will summarize the network for you')

        return parser.parse_args()

    def execute(self, n):
        while switch(n):
            if case('encoder'):
                netE = define_G(input_nc=1,
                                ngf=16, netG='encoder',
                                norm = 'batch', dropout_rate=self.dropout_rate,
                                init_type=self.init_type, init_gain=self.init_gain,
                                gpu_ids=self.gpu_ids) # a-OK
                summary(netE, input_size=(1, 256, 256))
                break

            if case('segmentor'):
                netC = define_C(input_nc=512, num_classes=self.num_classes,
                                netC='basic',
                                norm='none', dropout_rate=self.dropout_rate,
                                init_type=self.init_type, init_gain=0.01,
                                gpu_ids=self.gpu_ids) # a-OK

                summary(netC, input_size=(512, 32, 32))
                break

            if case('generator'):
                netG_T = define_G(input_nc=1, output_nc=1,
                                  ngf=32, netG='resnet_9blocks',
                                  norm='instance', dropout_rate=self.dropout_rate,
                                  init_type=self.init_type, init_gain=self.init_gain,
                                  gpu_ids=self.gpu_ids) # validate dimensions

                summary(netG_T, input_size=(1, 256, 256))
                break

            if case('decoder'):
                netU = define_G(input_nc=512, output_nc=1,
                                ngf=32, netG='decoder',
                                norm='instance', dropout_rate=self.dropout_rate,
                                init_type=self.init_type, init_gain=self.init_gain,
                                gpu_ids=self.gpu_ids) # validate dimensions

                summary(netU, input_size=[(512, 32, 32), (1, 256, 256)])
                break

            if case('discriminatorT'):
                netD_T = define_D(input_nc=1, ndf=64,
                                  netD='basic',
                                  norm='instance',
                                  init_type=self.init_type, init_gain=self.init_gain,
                                  gpu_ids=self.gpu_ids) # a-OK

                summary(netD_T, input_size=[(1, 256, 256)])
                break

            if case('discriminatorS'):
                netD_S = define_D(input_nc=1, ndf=64,
                                  netD='aux',
                                  norm='instance',
                                  init_type=self.init_type, init_gain=self.init_gain,
                                  gpu_ids=self.gpu_ids) # a-OK

                summary(netD_S, input_size=[(1, 256, 256)])
                break

            if case('discriminatorP'):
                netD_P = define_D(input_nc=5, ndf=64,
                                  netD='basic',
                                  norm='instance', init_type=self.init_type, init_gain=self.init_gain,
                                  gpu_ids=self.gpu_ids) # a-OK

                summary(netD_P, input_size=[(5, 256, 256)])
                break

            print ("Only above options are allowed.")
            break

if __name__ == "__main__":

    nt = NetTester()
    test = nt.parse()
    nt.execute(test.network)
