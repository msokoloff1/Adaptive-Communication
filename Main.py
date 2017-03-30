import argparse
import tensorflow as tf
import Networks as nets
import Utilities as utils
import numpy as np
from random import randint
import sys
import time

"""Parse command line args"""
parser = argparse.ArgumentParser()
parser.add_argument('-num_iters', default=22000, type=int, help="Sets the number of training iterations")
parser.add_argument('-message_length', default=120, type=int, help="Length of plaintext/ciphertext (Must be a multiple of 5)")
parser.add_argument('-batch_size', default=4096, type=int, help="Batch size used for training ops")
parser.add_argument('-optimizer', default='Adam', type=str,
                    help="Optimizer to be used when applying gradients (adam,adadelta,adagrad,rmsprop)")
parser.add_argument('-learning_rate', default=0.0008, type=int, help="Learning rate to be used when applying gradients")
parser.add_argument('-attack', default='none', type=str,help="Select attack type [\'none\',\'regular\',\'geometric\']")

args = parser.parse_args()
assert (args.message_length % 5 == 0), "Message length must be a multiple of 5"
"""Select optimizer"""
optimizer = tf.train.AdamOptimizer(args.learning_rate)
if (args.optimizer.lower() == 'adadelta'):
    optimizer = tf.train.AdadeltaOptimizer(args.learning_rate)

elif (args.optimizer.lower() == 'adagrad'):
    optimizer = tf.train.AdagradOptimizer(args.learning_rate)

elif (args.optimizer.lower() == 'rmsprop'):
    optimizer = tf.train.RMSPropOptimizer(args.learning_rate)

"""Instantiate nets"""
alice = nets.Encoder(args.message_length, 'aliceNet')
bob = nets.Decoder(args.message_length, alice, 'bobNet')
eve = nets.UnauthDecoder(args.message_length, alice, 'eveNet')

"""Calculate loss metrics"""
aliceAndBobLoss = utils.getBobAliceLoss(bob, eve, alice, args.message_length)
eveLoss = utils.getEveLoss(eve, alice)

"""Create generator for obtaining the correct update op"""
turnGen = utils.getTurn(alice.getUpdateOp(aliceAndBobLoss, optimizer)
                        , bob.getUpdateOp(aliceAndBobLoss, optimizer)
                        , eve.getUpdateOp(eveLoss, optimizer)
                        )

"""Instantiate logger, provide command line args for context"""
logger = utils.log(details=args._get_kwargs())

""" Generator for keeping track of key synchronization """
def expMovGen(logging= True):
	accuracyAB, accuracyC = 0.0, 0.0
	count = 0
	while(True):
		updateValueAB, updateValueC = yield [accuracyAB, accuracyC, count]
		if(logging):
			log(count, accuracyAB, accuracyC)

		accuracyAB = accuracyAB*0.99 + 0.01*updateValueAB
		accuracyC = accuracyC*0.99 + 0.01*updateValueC
		count += 1

"""Begin training loop"""


def train(numIters,sess):
    with tf.variable_scope("Train") as scope:
        start = time.time()
        #KEY MUST BE A DIFFERENT PART OF THE PARITY TREE EACH TIME. THE INDEX WILL BE MADE PUBLIC!
        accuracyManager = expMovGen(logging=False)
        accuracyManager.send(None)
        ABTreeSync = 0.0
        EveTreeSync = 0.0
        logMetrics = utils.getLoggingMetrics(bob, eve, alice)
        count = 0
        L = 10
        K = 100
        N = 32
        while(ABTreeSync< 0.99):
                inputs = np.random.randint(-L,L+1, [K,N])
                hiddenA, outputA = alice.tree.getActivations(inputs)
                hiddenB, outputB = bob.tree.getActivations(inputs)
                hiddenC, outputC = eve.tree.getActivations(inputs)
                alice.tree.updateWeights(inputs, hiddenA, outputA, outputB)
                bob.tree.updateWeights(inputs, hiddenB, outputB, outputA)
                eve.tree.updateWeights(inputs, hiddenC, outputA, int(outputB == outputC))
                ABTreeSync, EveTreeSync, _ = accuracyManager.send([float(outputA == outputB), float(outputA == outputB == outputC)])
                sys.stdout.write(
                    '\r' + "Key Gen Iteration : %d | A/B Accuracy : %f | C Accuracy : %f" % (count, ABTreeSync, EveTreeSync))

                count+= 1
        print("A/B Accuracy : %f | C Accuracy : %f" % (ABTreeSync, EveTreeSync))

        if(EveTreeSync > 0.5):
            exit(0)


        dataGen = utils.getData(args.message_length, args.batch_size)

        for iter in range(args.num_iters):
            data = next(dataGen)
            sys.stdout.write('\r' + "Iteration : %d"%(iter))
            sys.stdout.flush()

            feedDict = {
                 alice._inputKey: alice.tree.getKey(args.message_length, args.batch_size, iter)
                , bob._inputKey: bob.tree.getKey(args.message_length, args.batch_size,iter)
                , eve._inputKey: eve.tree.getKey(args.message_length, args.batch_size, iter)
                , alice._inputMessage: np.array(data['plainText'])
            }
            updateOps = next(turnGen)
            sess.run(updateOps, feed_dict=feedDict)
            if (iter % 100 == 0):
                aliceAndBobLossEvaluated, eveLossEvaluated, eveIncorrect, bobIncorrect = sess.run(
                    [tf.reduce_mean(aliceAndBobLoss), tf.reduce_mean(eveLoss)] + logMetrics, feed_dict=feedDict)
                logger.writeToFile([iter, aliceAndBobLossEvaluated, eveLossEvaluated, eveIncorrect, bobIncorrect])
                print("Iteration %s | Alice/Bob Loss : %g  | Eve Incorrect : %g | Bob Incorrect : %g" % (
                str(iter).zfill(6), aliceAndBobLossEvaluated, eveIncorrect, bobIncorrect))

            if(iter == (args.num_iters-1)):
                print("Training Took %s"%str(time.time() - start))
                return sess.run(logMetrics, feed_dict=feedDict)



def test(sess):
    with tf.variable_scope("Test") as scope:
        dataGen = utils.getData(args.message_length, args.batch_size)
        logMetrics = utils.getLoggingMetrics(bob, eve, alice)
        for testIter in range(50000):
            data = next(dataGen)
            N = 32
            K = 100
            L = 10
            feedDict = {
                alice._inputKey: alice.tree.getKey(args.message_length, args.batch_size,testIter)
                , bob._inputKey: bob.tree.getKey(args.message_length, args.batch_size,testIter)
                , eve._inputKey: eve.tree.getKey(args.message_length, args.batch_size,testIter)
                , alice._inputMessage: np.array(data['plainText'])
            }
            sess.run(eve.getUpdateOp(eveLoss, optimizer), feed_dict=feedDict)
            if (testIter % 100 == 0):
                aliceAndBobLossEvaluated, eveLossEvaluated, eveIncorrect, bobIncorrect = sess.run(
                    [tf.reduce_mean(aliceAndBobLoss), tf.reduce_mean(eveLoss)] + logMetrics, feed_dict=feedDict)
                logger.writeToFile([testIter, aliceAndBobLossEvaluated, eveLossEvaluated, eveIncorrect, bobIncorrect])
                print(
                    "Iteration %s | Alice/Bob Loss : %g  | Eve Incorrect : %g | Bob Incorrect : %g" % (
                        str(testIter).zfill(6), aliceAndBobLossEvaluated, eveIncorrect, bobIncorrect))


with tf.Session() as sess:
    saver = tf.train.Saver()    
    path = "savedModels/successfulModel.ckpt"
    try:
        saver.restore(sess, path)
        print("Successfully Restored Model!!")
    except:
        print("No model available for restoration")
        sess.run(tf.initialize_all_variables())

    eWrong, bWrong = train(args.num_iters, sess)
    #if(bWrong < 0.1 and eWrong > 0.2):
    saver.save(sess, path)
    test(sess)

