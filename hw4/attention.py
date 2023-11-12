import torch
import numpy as np

class Softmax:

    '''
    DO NOT MODIFY! AN INSTANCE IS ALREADY SET IN THE Attention CLASS' CONSTRUCTOR. USE IT!
    Performs softmax along the last dimension
    '''
    def forward(self, Z):

        z_original_shape = Z.shape

        self.N = Z.shape[0]*Z.shape[1]
        self.C = Z.shape[2]
        Z = Z.reshape(self.N, self.C)

        Ones_C = torch.ones((self.C, 1))
        self.A = torch.exp(Z) / (torch.exp(Z) @ Ones_C)

        return self.A.reshape(z_original_shape)

    def backward(self, dLdA):

        dLdA_original_shape = dLdA.shape

        dLdA = dLdA.reshape(self.N, self.C)

        dLdZ = torch.zeros((self.N, self.C))

        for i in range(self.N):

            J = torch.zeros((self.C, self.C))

            for m in range(self.C):
                for n in range(self.C):
                    if n == m:
                        J[m, n] = self.A[i][m] * (1 - self.A[i][m])
                    else:
                        J[m, n] = -self.A[i][m] * self.A[i][n]

            dLdZ[i, :] = dLdA[i, :] @ J

        return dLdZ.reshape(dLdA_original_shape)

class Attention:

        def __init__(self, weights_keys, weights_queries, weights_values):

            """
            Initialize instance variables. Refer to writeup for notation.
            input_dim = D, key_dim = query_dim = D_k, value_dim = D_v

            Argument(s)
            -----------

            weights_keys (torch.tensor, dim = (D X D_k)): weight matrix for keys
            weights_queries (torch.tensor, dim = (D X D_k)): weight matrix for queries
            weights_values (torch.tensor, dim = (D X D_v)): weight matrix for values

            """

            # Store the given weights as parameters of the class.
            self.W_k    = weights_keys
            self.W_q    = weights_queries
            self.W_v    = weights_values

            # Use this object to perform softmax related operations.
            # It performs softmax over the last dimension which is what you'll need.
            self.softmax = Softmax()

        def forward(self, X):

            """
            Compute outputs of the self-attention layer.
            Stores keys, queries, values, raw and normalized attention weights.
            Refer to writeup for notation.
            batch_size = B, seq_len = T, input_dim = D, value_dim = D_v

            Note that input to this method is a batch not a single sequence, so doing a transpose using .T can yield unexpected results.
            You should permute only the required axes.

            Input
            -----
            X (torch.tensor, dim = (B, T, D)): Input batch

            Return
            ------
            X_new (torch.tensor, dim = (B, T, D_v)): Output batch

            """

            self.X = X

            # Compute the values of Key, Query and Value

            self.Q = X @ self.W_q # (B, T, D_k)
            self.K = X @ self.W_k # (B, T, D_k)
            self.V = X @ self.W_v # (B, T, D_v)

            # Calculate Attention weights

            self.A_w    = self.Q @ self.K.transpose(1, 2) # (B, T, T)
            self.A_sig   = self.softmax.forward(self.A_w / np.sqrt(self.Q.shape[2]))

            # Calculate Attention context

            X_new         = self.A_sig @ self.V

            return X_new

        def backward(self, dLdXnew):

            """
            Backpropogate derivatives through the self-attention layer.
            Stores derivatives wrt keys, queries, values, and weight matrices.
            Refer to writeup for notation.
            batch_size = B, seq_len = T, input_dim = D, value_dim = D_v

            Note that input to this method is a batch not a single sequence, so doing a transpose using .T can yield unexpected results.
            You should permute only the required axes.

            Input
            -----
            dLdXnew (torch.tensor, dim = (B, T, D_v)): Derivative of the divergence wrt attention layer outputs

            Return
            ------
            dLdX (torch.tensor, dim = (B, T, D)): Derivative of the divergence wrt attention layer inputs

            """

            # Derivatives wrt attention weights (raw and normalized)

            dLdA_sig       = dLdXnew @ self.V.transpose(1, 2) # (B, T, T)
            dLdA_w         = self.softmax.backward(dLdA_sig) / np.sqrt(self.Q.shape[2]) # (B, T, T)

            # Derivatives wrt keys, queries, and value

            self.dLdV      = self.A_sig.transpose(1, 2) @ dLdXnew # (B, T, D_v)
            self.dLdK      = dLdA_w.transpose(1, 2) @ self.Q # (B, T, D_k)
            self.dLdQ      = dLdA_w @ self.K # (B, T, D_k)

            # Dervatives wrt weight matrices
            # Remember that you need to sum the derivatives along the batch dimension.

            self.dLdWq = (self.X.transpose(1, 2) @ self.dLdQ).sum(axis=0)
            self.dLdWv = (self.X.transpose(1, 2) @ self.dLdV).sum(axis=0)
            self.dLdWk = (self.X.transpose(1, 2) @ self.dLdK).sum(axis=0)

            # Derivative wrt input

            dLdX      = self.dLdV @ self.W_v.T + self.dLdK @ self.W_k.T + self.dLdQ @ self.W_q.T

            return dLdX
