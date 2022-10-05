import os
import numpy as np
import tensorflow as tf

from recommenders.models.deeprec.models.sequential.sli_rec import SLI_RECModel
from recommenders.models.deeprec.models.sequential.asvd import A2SVDModel


class SLI_RECModel_Custom(SLI_RECModel): 
    def fit(
        self,
        train_file,
        valid_file,
        valid_num_ngs,
        eval_metric="group_auc",
    ):
        """Fit the model with `train_file`. Evaluate the model on `valid_file` per epoch to observe the training status.
        If `test_file` is not None, evaluate it too.

        Args:
            train_file (str): training data set.
            valid_file (str): validation set.
            valid_num_ngs (int): the number of negative instances with one positive instance in validation data.
            eval_metric (str): the metric that control early stopping. e.g. "auc", "group_auc", etc.

        Returns:
            object: An instance of self.
        """

        # check bad input.
        if not self.need_sample and self.train_num_ngs < 1:
            raise ValueError(
                "Please specify a positive integer of negative numbers for training without sampling needed."
            )
        if valid_num_ngs < 1:
            raise ValueError(
                "Please specify a positive integer of negative numbers for validation."
            )

        if self.need_sample and self.train_num_ngs < 1:
            self.train_num_ngs = 1

        if self.hparams.write_tfevents and self.hparams.SUMMARIES_DIR:
            if not os.path.exists(self.hparams.SUMMARIES_DIR):
                os.makedirs(self.hparams.SUMMARIES_DIR)

            self.writer = tf.compat.v1.summary.FileWriter(
                self.hparams.SUMMARIES_DIR, self.sess.graph
            )

        train_sess = self.sess
        train_info, eval_info = [], []
        
        best_metric, self.best_epoch = 0, 0

        for epoch in range(1, self.hparams.epochs + 1):
            step = 0
            self.hparams.current_epoch = epoch
            epoch_loss = 0
            file_iterator = self.iterator.load_data_from_file(
                train_file,
                min_seq_length=self.min_seq_length,
                batch_num_ngs=self.train_num_ngs,
            )
            
            step_loss_list, step_data_loss_list = [], []
            for batch_data_input in file_iterator:
                if batch_data_input:
                    step_result = self.train(train_sess, batch_data_input)
                    (_, _, step_loss, step_data_loss, summary) = step_result
                    if self.hparams.write_tfevents and self.hparams.SUMMARIES_DIR:
                        self.writer.add_summary(summary, step)
                    epoch_loss += step_loss
                    
                    step_loss_list.append(step_loss)
                    step_data_loss_list.append(step_data_loss)
                    
                    step += 1
                    if step % self.hparams.show_step == 0:
                        print(
                            "step {0:d} , total_loss: {1:.4f}, data_loss: {2:.4f}".format(
                                step, step_loss, step_data_loss
                            )
                        )
            
            avg_step_loss = np.mean(step_loss_list) 
            avg_step_data_loss = np.mean(step_data_loss)
            
            train_res = {"loss": avg_step_loss, "data_loss": avg_step_data_loss}
            train_info.append((epoch, train_res)) 
            
            valid_res = self.run_eval(valid_file, valid_num_ngs)
            print(
                "eval valid at epoch {0}: {1}".format(
                    epoch,
                    ",".join(
                        [
                            "" + str(key) + ":" + str(value)
                            for key, value in valid_res.items()
                        ]
                    ),
                )
            )
            eval_info.append((epoch, valid_res))

            progress = False
            early_stop = self.hparams.EARLY_STOP
            if valid_res[eval_metric] > best_metric:
                best_metric = valid_res[eval_metric]
                self.best_epoch = epoch
                progress = True
            else:
                if early_stop > 0 and epoch - self.best_epoch >= early_stop:
                    print("early stop at epoch {0}!".format(epoch))
                    break

            if self.hparams.save_model and self.hparams.MODEL_DIR:
                if not os.path.exists(self.hparams.MODEL_DIR):
                    os.makedirs(self.hparams.MODEL_DIR)
                if progress:
                    checkpoint_path = self.saver.save(
                        sess=train_sess,
                        save_path=self.hparams.MODEL_DIR + "epoch_" + str(epoch),
                    )
                    checkpoint_path = self.saver.save(  # noqa: F841
                        sess=train_sess,
                        save_path=os.path.join(self.hparams.MODEL_DIR, "best_model"),
                    )

        if self.hparams.write_tfevents:
            self.writer.close()

        print(eval_info)
        print("best epoch: {0}".format(self.best_epoch))
        return self, train_info, eval_info

class A2SVDModel_Custom(A2SVDModel): 
    def fit(
        self,
        train_file,
        valid_file,
        valid_num_ngs,
        eval_metric="group_auc",
    ):
        """Fit the model with `train_file`. Evaluate the model on `valid_file` per epoch to observe the training status.
        If `test_file` is not None, evaluate it too.

        Args:
            train_file (str): training data set.
            valid_file (str): validation set.
            valid_num_ngs (int): the number of negative instances with one positive instance in validation data.
            eval_metric (str): the metric that control early stopping. e.g. "auc", "group_auc", etc.

        Returns:
            object: An instance of self.
        """

        # check bad input.
        if not self.need_sample and self.train_num_ngs < 1:
            raise ValueError(
                "Please specify a positive integer of negative numbers for training without sampling needed."
            )
        if valid_num_ngs < 1:
            raise ValueError(
                "Please specify a positive integer of negative numbers for validation."
            )

        if self.need_sample and self.train_num_ngs < 1:
            self.train_num_ngs = 1

        if self.hparams.write_tfevents and self.hparams.SUMMARIES_DIR:
            if not os.path.exists(self.hparams.SUMMARIES_DIR):
                os.makedirs(self.hparams.SUMMARIES_DIR)

            self.writer = tf.compat.v1.summary.FileWriter(
                self.hparams.SUMMARIES_DIR, self.sess.graph
            )

        train_sess = self.sess
        train_info, eval_info = [], []
        
        best_metric, self.best_epoch = 0, 0

        for epoch in range(1, self.hparams.epochs + 1):
            step = 0
            self.hparams.current_epoch = epoch
            epoch_loss = 0
            file_iterator = self.iterator.load_data_from_file(
                train_file,
                min_seq_length=self.min_seq_length,
                batch_num_ngs=self.train_num_ngs,
            )
            
            step_loss_list, step_data_loss_list = [], []
            for batch_data_input in file_iterator:
                if batch_data_input:
                    step_result = self.train(train_sess, batch_data_input)
                    (_, _, step_loss, step_data_loss, summary) = step_result
                    if self.hparams.write_tfevents and self.hparams.SUMMARIES_DIR:
                        self.writer.add_summary(summary, step)
                    epoch_loss += step_loss
                    
                    step_loss_list.append(step_loss)
                    step_data_loss_list.append(step_data_loss)
                    
                    step += 1
                    if step % self.hparams.show_step == 0:
                        print(
                            "step {0:d} , total_loss: {1:.4f}, data_loss: {2:.4f}".format(
                                step, step_loss, step_data_loss
                            )
                        )
            
            avg_step_loss = np.mean(step_loss_list) 
            avg_step_data_loss = np.mean(step_data_loss)
            
            train_res = {"loss": avg_step_loss, "data_loss": avg_step_data_loss}
            train_info.append((epoch, train_res)) 
            
            valid_res = self.run_eval(valid_file, valid_num_ngs)
            print(
                "eval valid at epoch {0}: {1}".format(
                    epoch,
                    ",".join(
                        [
                            "" + str(key) + ":" + str(value)
                            for key, value in valid_res.items()
                        ]
                    ),
                )
            )
            eval_info.append((epoch, valid_res))

            progress = False
            early_stop = self.hparams.EARLY_STOP
            if valid_res[eval_metric] > best_metric:
                best_metric = valid_res[eval_metric]
                self.best_epoch = epoch
                progress = True
            else:
                if early_stop > 0 and epoch - self.best_epoch >= early_stop:
                    print("early stop at epoch {0}!".format(epoch))
                    break

            if self.hparams.save_model and self.hparams.MODEL_DIR:
                if not os.path.exists(self.hparams.MODEL_DIR):
                    os.makedirs(self.hparams.MODEL_DIR)
                if progress:
                    checkpoint_path = self.saver.save(
                        sess=train_sess,
                        save_path=self.hparams.MODEL_DIR + "epoch_" + str(epoch),
                    )
                    checkpoint_path = self.saver.save(  # noqa: F841
                        sess=train_sess,
                        save_path=os.path.join(self.hparams.MODEL_DIR, "best_model"),
                    )

        if self.hparams.write_tfevents:
            self.writer.close()

        print(eval_info)
        print("best epoch: {0}".format(self.best_epoch))
        return self, train_info, eval_info

