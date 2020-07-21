from tensorflow import reduce_sum, greater, reduce_mean, square, constant, cast, float32


def orientation_loss(y_true, y_pred):
    """
    input: y_true -- (batch_size, bin, 2) ground truth orientation value in cos and sin form.
           y_pred -- (batch_size, bin ,2) estimated orientation value from the ConvNet
    output: loss -- loss values for orientation
    """

    # sin^2 + cons^2
    anchors = reduce_sum(square(y_true), axis=2)
    # check which bin valid
    anchors = greater(anchors, constant(0.5))
    # add valid bin
    anchors = reduce_sum(cast(anchors, float32), 1)

    # cos(true)cos(estimate) + sin(true)sin(estimate)
    loss = y_true[:, :, 0] * y_pred[:, :, 0] + y_true[:, :, 1] * y_pred[:, :, 1]
    # the mean value in each bin
    loss = reduce_sum(loss, axis=1) / anchors
    # sum the value at each bin
    loss = reduce_mean(loss)
    loss = 2 - 2 * loss

    return loss
