import numpy as np
import torch


_n_lookahead = 50
_n_burn_in = 500


# class DynamicNet(torch.nn.Module):
#     def __init__(self, D_in, H, D_out):
#         super(DynamicNet, self).__init__()
#         self.input_linear = torch.nn.Linear(D_in, H)
#         self.middle_linear = torch.nn.Linear(H, H)
#         self.output_linear = torch.nn.Linear(H, D_out)

#     def forward(self, x):
#         h_relu = self.input_linear(x).clamp(min=0)
#         for _ in range(5):
#             h_relu = self.middle_linear(h_relu).clamp(min=0)
#         y_pred = self.output_linear(h_relu)
#         return y_pred
class Propagate(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input


class Simulation(torch.nn.Module):
    def __init__(self, n_var=6):
        # The variables
        self.order = np.ones(shape=(n_var, ))
        self.mask = np.zeros(shape=(n_var, ))
        self.c = torch.randn(n_var, requires_grad=True)
        self.unit = np.ones(shape=(n_var,))

    def freeze_parameters(self,
                          mask=np.array([1, 1, 1,
                                         0, 0, 1])):
        self.mask = mask

    def assign_parameters(self,
                          pars=np.array([0, 50,
                                         0., np.sqrt(0.02),
                                         1., 2.])):
        print("pars : ", pars)
        print("mask : ", self.mask)
        self.c.assign(self.c * (self.unit - self.mask) +
                      pars * self.mask)
        n_max = 500
        self.n = n_max

    def force(self, x):
        # binary interactions defined here
        f = np.zeros(shape=x.shape)
        # for i, p in enumerate(x):
        #     f[i, :] = 0
        #     for ii, pp in enumerate(x):
        #         rel_pos = p - pp
        #         f[i] += - self.g * \
        #             rel_pos / pow(np.linalg.norm(rel_pos),
        #                           1. + self.k)
        f += - self.g * \
            x / pow(np.linalg.norm(x),
                    1. + self.k)
        return f

    def transform(self, x):
        xx = x[1] * np.array([np.cos(x[0]) - np.sin(x[0]),
                              np.cos(x[0]) + np.sin(x[0])])
        return xx

    def propagate(self, x, v):
        x += v
        v += self.force(x)
        return x

    def inverse_transform(self, x):
        xx = np.array([np.arctan2(x[1], x[0]),
                       np.sqrt(np.dot(x, x))])
        return xx

    def forward(self, times):
        # The formula

        self.phi = self.c[0]
        self.r = self.c[1]

        self.v = np.array([self.c[2], self.c[3]])
        self.g = self.c[4]
        self.k = self.c[5]

        self.r0 = self.r / 2
        self.v0 = self.v / np.sqrt(2)

        output = []
        x = self.transform([self.phi, self.r])
        x0 = self.transform([self.phi, self.r0])

        for step in times:
            x = self.propagate(x, self.v)
            x0 = self.propagate(x0, self.v0)
            output.append(self.inverse_transform(x - x0)[0])
        return output


x = range(100)
y = np.mod(x, (2. * np.pi))

model = Simulation()
learning_rate = 1e-4

for t in range(500):
    y_pred = model.forward(x)

    # Compute and print loss
    loss = model.loss(y_pred, y)
    print(t, loss.item())

    # Use autograd to compute the backward pass.
    loss.backward()

    # Update weights using gradient descent
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad











