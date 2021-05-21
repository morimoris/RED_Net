from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Conv2D, Conv2DTranspose, Input, Add, ReLU

class RED_Net():
    def __init__(self, conv_num):
        #parameter set
        self.conv_num = conv_num
        self.deconv_num = conv_num    #Number of convolution layers == number of deconvolutions layers
        self.input_channels = 1       #input gray scale images
        self.filter_num = 64
        self.kernel_size = (3, 3)

        #create a list to store skip connections
        if self.conv_num % 2 == 1:
            self.skip_connection_list = (self.conv_num // 2 + 1) * [None]
        else:
            self.skip_connection_list = (self.conv_num // 2) * [None]

    def RED_Net_odd(self):
        """
        if conv_num % 2 == 1
        """
        input_shape = Input((None, None, self.input_channels))
        self.skip_connection_list[0] = input_shape
        
        #convolution part
        conv = input_shape
        for i in range(self.conv_num):
            conv = Conv2D(filters = self.filter_num, kernel_size = self.kernel_size, padding = "same")(conv)

            if i % 2 == 1:
                self.skip_connection_list[(i // 2 + 1)] = conv

        #deconvolution part
        for i in range(self.deconv_num - 1):
            conv = Conv2DTranspose(filters = self.filter_num, kernel_size = self.kernel_size, padding = "same")(conv)
            
            #add skip connections
            if i % 2 == 0:
                deconv_skip = Add()([conv, self.skip_connection_list[-1 * (i // 2 + 1)]])
                conv = ReLU()(deconv_skip)

        conv = Conv2DTranspose(filters = self.input_channels, kernel_size = self.kernel_size, padding = "same")(conv)
        #add skip connections
        deconv_skip = Add()([conv, self.skip_connection_list[0]])
        Output = ReLU()(deconv_skip)

        model = Model(inputs = input_shape, outputs = Output)
        model.summary()

        return model

    def RED_Net_even(self):
        """
        if conv_num % 2 == 0
        """
        input_shape = Input((None, None, self.input_channels))
        self.skip_connection_list[0] = input_shape
        
        #convolution part   
        conv = input_shape
        for i in range(self.conv_num):
            conv = Conv2D(filters = self.filter_num, kernel_size = self.kernel_size, padding = "same")(conv)

            if i % 2 == 1 and i != self.conv_num - 1:
                self.skip_connection_list[(i // 2 + 1)] = conv

        #deconvolution part
        for i in range(self.deconv_num - 1):
            conv = Conv2DTranspose(filters = self.filter_num, kernel_size = self.kernel_size, padding = "same")(conv)
            
            #add skip connections
            if i % 2 == 1:
                deconv_skip = Add()([conv, self.skip_connection_list[-1 * (i // 2 + 1)]])
                conv = ReLU()(deconv_skip)

        conv = Conv2DTranspose(filters = self.input_channels, kernel_size = self.kernel_size, padding = "same")(conv)
        #add skip connections
        deconv_skip = Add()([conv, self.skip_connection_list[0]])
        Output = ReLU()(deconv_skip)

        model = Model(inputs = input_shape, outputs = Output)
        model.summary()
        return model
