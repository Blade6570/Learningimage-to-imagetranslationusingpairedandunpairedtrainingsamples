import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from data.supervised_dataset import supervisedDataset
opt = TrainOptions().parse()

data_loader = CreateDataLoader(opt,0)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
super_data_load= opt.super_start
if  super_data_load==1:
    data_loader_super = CreateDataLoader(opt,1)
    dataset_super = data_loader_super.load_data()
    dataset_super_size = len(data_loader_super)
    print('#training images = %d' % dataset_super_size)

if opt.super_epoch_start>0:
   opt.super_start=0

model = create_model(opt)
visualizer = Visualizer(opt)
total_steps = 0
#supervised_train=opt.super_epoch_start

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    epoch_iter = 0
#    opt.super_start=0
    if epoch>=opt.super_epoch_start and epoch <= opt.super_epoch_start+opt.super_epoch and super_data_load==1:
       opt.super_start=1
       for i, data in enumerate(dataset_super):
           iter_start_time = time.time()
           visualizer.reset()
           total_steps += opt.batchSize
           epoch_iter += opt.batchSize
           model.set_input(data)
           model.optimize_parameters(opt)

           if total_steps % opt.display_freq == 0:
              save_result = total_steps % opt.update_html_freq == 0
              visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

           if total_steps % opt.print_freq == 0:
              errors = model.get_current_errors()
              t = (time.time() - iter_start_time) / opt.batchSize
              visualizer.print_current_errors(epoch, epoch_iter, errors, t)
              if opt.display_id > 0:
                 visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

           if total_steps % opt.save_latest_freq == 0:
               print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
               model.save('latest')
    
       
     
        
    else:
       opt.super_start=0
       for i, data in enumerate(dataset):
           iter_start_time = time.time()
           visualizer.reset()
           total_steps += opt.batchSize
           epoch_iter += opt.batchSize
           model.set_input(data)
           model.optimize_parameters(opt)

           if total_steps % opt.display_freq == 0:
              save_result = total_steps % opt.update_html_freq == 0
              visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

           if total_steps % opt.print_freq == 0:
              errors = model.get_current_errors()
              t = (time.time() - iter_start_time) / opt.batchSize
              visualizer.print_current_errors(epoch, epoch_iter, errors, t)
              if opt.display_id > 0:
                 visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

           if total_steps % opt.save_latest_freq == 0:
               print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
               model.save('latest')
   # if epoch>=supervised_train+25:
    #   supervised_train=supervised_train+50

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    model.update_learning_rate()
