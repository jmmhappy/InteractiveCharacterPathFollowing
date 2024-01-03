import torch
from torch.utils.checkpoint import checkpoint
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os

class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        HIDDEN = 30
        self.gru = torch.nn.GRU(input_size=4, hidden_size=HIDDEN, batch_first=True, bidirectional=True).float()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(2*HIDDEN, 2), # 2 for bidirectional
            torch.nn.Tanh() # output (-1,1)
        ).float()

        # initial hidden state for each element in the batch
        self.h0 = torch.nn.Parameter(torch.zeros(2*self.gru.num_layers, 1, HIDDEN).normal_(std=0.01), requires_grad=True)

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        if len(x.shape) == 2: # batch length(=1) not included(runtime)
            x = x.unsqueeze(0) # make it three
        
        h0 = self.h0.expand(-1, x.shape[0], -1).contiguous()
        out, h = self.gru(x.float(), h0)
#        print(h)
        out = self.fc(out)
        return out

    def sample(self, start, given, window):
        hidden_stack = []

        x = torch.tensor(start).unsqueeze(1).float() # size = (seq, batch, input size)
        hprev = self.h0.expand(-1, x.shape[0], -1).contiguous()

        gru_out, h = self.gru(x, hprev)
        hidden_stack = gru_out

        out = self.fc(hidden_stack)

        for p in given: # 2dim
            p = torch.tensor([p])
            p = torch.cat((out.squeeze(), p)).unsqueeze(1) # make 4
            gru_out, h = self.gru(p.float(), hprev)
            out = self.fc(hidden_stack)

            hidden_stack = torch.cat((hidden_stack, gru_out), axis=1)[:,-window:,:]

            hprev = h

        out = self.fc(hidden_stack)
        return out.squeeze(0).cpu().detach().numpy() # direction of the most recent max 120 points



class BiGRUNetwork:
    def __init__(self):
        self.model = SimpleModel()
        self.use_cuda = False

        self.optimizer = torch.optim.Adam(self.model.parameters())

    def cuda(self):
        if torch.cuda.is_available():
            print('cuda is available')
            self.model.cuda()
            self.use_cuda = True
            return True
        return False

    def predict(self, obs, window): 
        self.model.eval() # eval mode

        pastPosDir, futurePos = obs

#        directions = self.model.sample(pastPosDir, futurePos, window)

        indices = [0]
        for i in [9,19,29]:
            if i < len(futurePos):
                indices.append(i)

        x = torch.tensor(pastPosDir).unsqueeze(1).float()
        directions = []
        for i, point in enumerate(futurePos):
            d = self.model.forward(x)[-1]
            f = torch.cat((torch.tensor([point]), d), 1)
            x = torch.stack((x[-1,:,:], f))
            if i in indices:
                directions.append(d.squeeze().cpu().detach().numpy())

#        for point in future:
#            out = self.model(seq)
#            out = out.squeeze(0) # single instance(batch len=1, so drop it)
#            out = out.cpu().detach().numpy()
#            seq.append(np.concatenate((point, out[-1])))
#
#        directions = np.array(seq)[logLen:, 2:]

        for i in range(len(directions)):
            directions[i] /= np.sqrt(sum(directions[i] * directions[i]))
        while len(directions) < 4:
            directions.append(directions[-1])
        return directions

    def batch_preparation(self, train_data, batch_size):

#        POINTS = len(train_data[0][0]) # window size 

#        batch_in = np.zeros((batch_size, POINTS, 4), dtype='float32')
#        batch_out = np.zeros((batch_size, POINTS, 2), dtype='float32')

        # the data is already randomly sampled 
        train_data = np.array(train_data)
        data_in = train_data[:-1]
        data_out = train_data[1:, :, 2:] # only directions

        for i in range(0, len(train_data)-1, batch_size):
            batch_in, batch_out = data_in[i:i+batch_size], data_out[i:i+batch_size]
            yield batch_in, batch_out
           

    def train(self, save_name, train_data, batch_size=40, n_epoch=1000):
        
        model = self.model
        optimizer = self.optimizer

        loss_fn = torch.nn.L1Loss()

        writer = SummaryWriter('output/%s'%save_name)

        for epoch in range(n_epoch):

            batch_loss, N = 0, 0

            for x,y in self.batch_preparation(train_data, batch_size):
                x = torch.tensor(x, requires_grad=True)
                y = torch.from_numpy(y)
                if self.use_cuda:
                    x = x.cuda()
                    y = y.cuda()

                y_hat = model(x)
                # input size = (120,2), target size = (batch, 120, 2)
                loss = loss_fn(y_hat, y)

                # zero init
                optimizer.zero_grad()

                # save gradients
                loss.backward()

                # update model parameters
                optimizer.step()

                batch_loss += loss.item() * x.shape[0] 
                N += x.shape[0]
                
            batch_loss /= N
            print('Epoch: {:5d}, Loss: {:.5f}'.format(epoch+1, batch_loss))
            writer.add_scalar('training_loss', batch_loss, epoch)
            
            # save checkpoint
            if epoch % 100 == 0:
                self.save_weights('%s_ep_%d'%(save_name, epoch))

        print('Finished training')
        self.save_weights('%s_ep_%d'%(save_name, n_epoch))
        writer.close()


    def save_weights(self, filename):
        if not os.path.exists('output'):
            os.makedirs('output')

        torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                }, 'output/%s.bin' % filename)


    def load_weights(self, path):
        checkpoint = torch.load(path, map_location=lambda storage, loc:storage)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 

        # in case you use dropout layers or batch normalization
        # set your model into evaluation mode to make your result consistent
        # model.eval()
