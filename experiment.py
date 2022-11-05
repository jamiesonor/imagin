import os
import util
import random
import torch
import numpy as np
from model import *
from dataset import *
from tqdm import tqdm
from einops import repeat
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid


def step(model, criterion, dyn_v, dyn_a, sampling_endpoints, t, label, reg_lambda, clip_grad=0.0, device='cpu', optimizer=None, scheduler=None):
    if optimizer is None: model.eval()
    else: model.train()

    # run model
    logit, attention, latent, reg_ortho = model(dyn_v.to(device), dyn_a.to(device), t.to(device), sampling_endpoints)
    loss = criterion(logit, label.to(device))
    reg_ortho *= reg_lambda
    loss += reg_ortho

    # optimize model
    if optimizer is not None:
       optimizer.zero_grad()
       loss.backward()
       if clip_grad > 0.0: torch.nn.utils.clip_grad_value_(model.parameters(), clip_grad)
       optimizer.step()
       if scheduler is not None:
           scheduler.step()

    return logit, loss, attention, latent, reg_ortho


def train(argv):
    # make directories
    os.makedirs(os.path.join(argv.targetdir, 'model'), exist_ok=True)
    os.makedirs(os.path.join(argv.targetdir, 'summary'), exist_ok=True)
    os.makedirs(os.path.join(argv.targetdir, 'attention'), exist_ok=True)

    # set seed and device
    torch.manual_seed(argv.seed)
    np.random.seed(argv.seed)
    random.seed(argv.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(argv.seed)
    else:
        device = torch.device("cpu")

    # define dataset
    dataset = ABIDE(argv.sourcedir, roi=argv.roi, dynamic_length=argv.dynamic_length, k_fold=argv.k_fold, smoothing_fwhm=argv.fwhm)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=argv.minibatch_size, shuffle=False, num_workers=4, pin_memory=True)
    dataloader_test = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    logger_test = util.logger.LoggerIMAGIN(argv.k_fold, dataset.num_classes)

    # resume checkpoint if file exists
    if os.path.isfile(os.path.join(argv.targetdir, 'checkpoint.pth')):
        print('resuming checkpoint experiment')
        checkpoint = torch.load(os.path.join(argv.targetdir, 'checkpoint.pth'), map_location=device)
    else:
        checkpoint = {
            'fold': 0,
            'epoch': 0,
            'model': None,
            'optimizer': None,
            'scheduler': None}

    # start experiment
    for k in range(checkpoint['fold'], argv.k_fold):
        # make directories per fold
        os.makedirs(os.path.join(argv.targetdir, 'model', str(k)), exist_ok=True)

        # set dataloader
        dataset.set_fold(k, train=True)

        # define model
        model = ModelIMAGIN(
            input_dims=[dataset.aal_num_nodes, dataset.cc200_num_nodes, dataset.schaefer_num_nodes],
            hidden_dims=argv.hidden_dims,
            num_classes=2,
            num_layers=argv.num_layers,
            sparsities=argv.sparsities,
        )
        model.to(device)
        if checkpoint['model'] is not None: model.load_state_dict(checkpoint['model'])
        criterion = torch.nn.CrossEntropyLoss()

        # define optimizer and learning rate scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=argv.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=argv.max_lr, epochs=argv.num_epochs, steps_per_epoch=len(dataloader), pct_start=0.2, div_factor=argv.max_lr/argv.lr, final_div_factor=1000)
        if checkpoint['optimizer'] is not None: optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint['scheduler'] is not None: scheduler.load_state_dict(checkpoint['scheduler'])

        # define logging objects
        summary_writer = SummaryWriter(os.path.join(argv.targetdir, 'summary', str(k), 'train'), )
        summary_writer_val = SummaryWriter(os.path.join(argv.targetdir, 'summary', str(k), 'val'), )
        logger = util.logger.LoggerIMAGIN(argv.k_fold, dataset.num_classes)
        logger_val = util.logger.LoggerIMAGIN(argv.k_fold, dataset.num_classes)

        best_accuracy = 0
        best_model = None

        # start training
        for epoch in range(checkpoint['epoch'], argv.num_epochs):
            logger.initialize(k)
            dataset.set_fold(k, train=True)
            loss_accumulate = 0.0
            reg_ortho_accumulate = 0.0
            for i, x in enumerate(tqdm(dataloader, ncols=60, desc=f'k:{k} e:{epoch}')):
                # process input data
                dyn_a, sampling_points = util.bold.process_dynamic_fc(x['timeseries'], argv.window_size, argv.window_stride, argv.dynamic_length)
                sampling_endpoints = [p+argv.window_size for p in sampling_points]
                if i==0: dyn_v = repeat(torch.eye(dataset.num_nodes), 'n1 n2 -> b t n1 n2', t=len(sampling_points), b=argv.minibatch_size)
                if len(dyn_a) < argv.minibatch_size: dyn_v = dyn_v[:len(dyn_a)]
                t = x['timeseries'].permute(1,0,2)
                label = x['label']

                logit, loss, attention, latent, reg_ortho = step(
                    model=model,
                    criterion=criterion,
                    dyn_v=dyn_v,
                    dyn_a=dyn_a,
                    sampling_endpoints=sampling_endpoints,
                    t=t,
                    label=label,
                    reg_lambda=argv.reg_lambda,
                    clip_grad=argv.clip_grad,
                    device=device,
                    optimizer=optimizer,
                    scheduler=scheduler)
                pred = logit.argmax(1)
                prob = logit.softmax(1)
                if np.isnan(prob.detach().cpu().numpy()).any():
                    print('nan')
                loss_accumulate += loss.detach().cpu().numpy()
                reg_ortho_accumulate += reg_ortho.detach().cpu().numpy()
                logger.add(k=k, pred=pred.detach().cpu().numpy(), true=label.detach().cpu().numpy(), prob=prob.detach().cpu().numpy())
                summary_writer.add_scalar('lr', scheduler.get_last_lr()[0], i+epoch*len(dataloader))

            # summarize results
            samples = logger.get(k)
            metrics = logger.evaluate(k)
            summary_writer.add_scalar('loss', loss_accumulate/len(dataloader), epoch)
            summary_writer.add_scalar('reg_ortho', reg_ortho_accumulate/len(dataloader), epoch)
            summary_writer.add_pr_curve('precision-recall', samples['true'], samples['prob'][:,1], epoch)
            [summary_writer.add_scalar(key, value, epoch) for key, value in metrics.items() if not key=='fold']
            [summary_writer.add_image(key, make_grid(value[-1].unsqueeze(1), normalize=True, scale_each=True), epoch) for key, value in attention.items()]
            summary_writer.flush()
            print("TRAIN:", metrics)

            # save checkpoint
            torch.save({
                'fold': k,
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()},
                os.path.join(argv.targetdir, 'checkpoint.pth'))

            # check validation results
            logger_val.initialize(k)
            dataset.set_fold(k, train=False)
            loss_accumulate = 0.0
            reg_ortho_accumulate = 0.0
            for i, x in enumerate(tqdm(dataloader_test, ncols=60, desc=f'k:{k} e:{epoch}')):
                with torch.no_grad():
                    # process input data
                    dyn_a, sampling_points = util.bold.process_dynamic_fc(x['timeseries'], argv.window_size,
                                                                          argv.window_stride)
                    sampling_endpoints = [p + argv.window_size for p in sampling_points]
                    if i == 0: dyn_v = repeat(torch.eye(dataset.num_nodes), 'n1 n2 -> b t n1 n2',
                                              t=len(sampling_points), b=argv.minibatch_size)
                    if not dyn_v.shape[1] == dyn_a.shape[1]: dyn_v = repeat(torch.eye(dataset.num_nodes),
                                                                            'n1 n2 -> b t n1 n2',
                                                                            t=len(sampling_points),
                                                                            b=argv.minibatch_size)
                    if len(dyn_a) < argv.minibatch_size: dyn_v = dyn_v[:len(dyn_a)]
                    t = x['timeseries'].permute(1, 0, 2)
                    label = x['label']

                    logit, loss, attention, latent, reg_ortho = step(
                        model=model,
                        criterion=criterion,
                        dyn_v=dyn_v,
                        dyn_a=dyn_a,
                        sampling_endpoints=sampling_endpoints,
                        t=t,
                        label=label,
                        reg_lambda=argv.reg_lambda,
                        clip_grad=argv.clip_grad,
                        device=device,
                        optimizer=None,
                        scheduler=None)
                    pred = logit.argmax(1)
                    prob = logit.softmax(1)
                    logger_val.add(k=k, pred=pred.detach().cpu().numpy(), true=label.detach().cpu().numpy(),
                               prob=prob.detach().cpu().numpy())
                    loss_accumulate += loss.detach().cpu().numpy()
                    reg_ortho_accumulate += reg_ortho.detach().cpu().numpy()


            # summarize results
            samples = logger_val.get(k)
            metrics = logger_val.evaluate(k)
            summary_writer_val.add_scalar('loss', loss_accumulate / len(dataloader_test), epoch)
            summary_writer_val.add_scalar('reg_ortho', reg_ortho_accumulate / len(dataloader_test), epoch)
            summary_writer_val.add_pr_curve('precision-recall', samples['true'], samples['prob'][:, 1], epoch)
            [summary_writer_val.add_scalar(key, value, epoch) for key, value in metrics.items() if not key == 'fold']
            [summary_writer_val.add_image(key, make_grid(value[-1].unsqueeze(1), normalize=True, scale_each=True), epoch) for
             key, value in attention.items()]
            summary_writer_val.flush()
            print("VAL:", metrics)

            if metrics['accuracy'] > best_accuracy or (metrics['accuracy'] == best_accuracy and metrics['roc_auc'] > best_auroc):
                best_accuracy = metrics['accuracy']
                best_auroc = metrics['roc_auc']
                best_model = model.state_dict()
                print("BEST MODEL")

        # finalize fold
        torch.save(best_model, os.path.join(argv.targetdir, 'model', str(k), 'model.pth'))
        checkpoint.update({'epoch': 0, 'model': None, 'optimizer': None, 'scheduler': None})

        # final validation results
        os.makedirs(os.path.join(argv.targetdir, 'attention', str(k)), exist_ok=True)

        model.load_state_dict(torch.load(os.path.join(argv.targetdir, 'model', str(k), 'model.pth')))

        # define logging objects
        fold_attention = {'node_attention': [], 'time_attention': []}
        summary_writer_test = SummaryWriter(os.path.join(argv.targetdir, 'summary', str(k), 'test'))

        logger_test.initialize(k)
        dataset.set_fold(k, train=False)
        loss_accumulate = 0.0
        reg_ortho_accumulate = 0.0
        latent_accumulate = []
        for i, x in enumerate(tqdm(dataloader_test, ncols=60, desc=f'k:{k}')):
            with torch.no_grad():
                # process input data
                dyn_a, sampling_points = util.bold.process_dynamic_fc(x['timeseries'], argv.window_size,
                                                                      argv.window_stride)
                sampling_endpoints = [p + argv.window_size for p in sampling_points]
                if i == 0: dyn_v = repeat(torch.eye(dataset.num_nodes), 'n1 n2 -> b t n1 n2',
                                          t=len(sampling_points), b=argv.minibatch_size)
                if not dyn_v.shape[1] == dyn_a.shape[1]: dyn_v = repeat(torch.eye(dataset.num_nodes),
                                                                        'n1 n2 -> b t n1 n2',
                                                                        t=len(sampling_points),
                                                                        b=argv.minibatch_size)
                if len(dyn_a) < argv.minibatch_size: dyn_v = dyn_v[:len(dyn_a)]
                t = x['timeseries'].permute(1, 0, 2)
                label = x['label']

                logit, loss, attention, latent, reg_ortho = step(
                    model=model,
                    criterion=criterion,
                    dyn_v=dyn_v,
                    dyn_a=dyn_a,
                    sampling_endpoints=sampling_endpoints,
                    t=t,
                    label=label,
                    reg_lambda=argv.reg_lambda,
                    clip_grad=argv.clip_grad,
                    device=device,
                    optimizer=None,
                    scheduler=None)
                pred = logit.argmax(1)
                prob = logit.softmax(1)
                logger_test.add(k=k, pred=pred.detach().cpu().numpy(), true=label.detach().cpu().numpy(),
                           prob=prob.detach().cpu().numpy())
                loss_accumulate += loss.detach().cpu().numpy()
                reg_ortho_accumulate += reg_ortho.detach().cpu().numpy()

                fold_attention['node_attention'].append(attention['node-attention'].detach().cpu().numpy())
                fold_attention['time_attention'].append(attention['time-attention'].detach().cpu().numpy())
                latent_accumulate.append(latent.detach().cpu().numpy())

        # summarize results
        samples = logger_test.get(k)
        metrics = logger_test.evaluate(k)
        summary_writer_test.add_scalar('loss', loss_accumulate / len(dataloader))
        summary_writer_test.add_scalar('reg_ortho', reg_ortho_accumulate / len(dataloader))
        summary_writer_test.add_pr_curve('precision-recall', samples['true'], samples['prob'][:, 1])
        [summary_writer_test.add_scalar(key, value) for key, value in metrics.items() if not key == 'fold']
        [summary_writer_test.add_image(key, make_grid(value[-1].unsqueeze(1), normalize=True, scale_each=True)) for
         key, value in attention.items()]
        summary_writer_test.flush()
        print("FINAL VAL:", metrics)

        # finalize fold
        logger_test.to_csv(argv.targetdir, k)
        if argv.dataset == 'rest':
            [np.save(os.path.join(argv.targetdir, 'attention', str(k), f'{key}.npy'), np.concatenate(value)) for
             key, value in fold_attention.items()]
        np.save(os.path.join(argv.targetdir, 'attention', str(k), 'latent.npy'), np.concatenate(latent_accumulate))
        del fold_attention

    # finalize experiment
    logger_test.to_csv(argv.targetdir)
    final_metrics = logger_test.evaluate()
    print(final_metrics)
    torch.save(logger_test.get(), os.path.join(argv.targetdir, 'samples.pkl'))

    summary_writer.close()
    summary_writer_val.close()
    summary_writer_test.close()
    # os.remove(os.path.join(argv.targetdir, 'checkpoint.pth'))


def test(argv):
    os.makedirs(os.path.join(argv.targetdir, 'attention'), exist_ok=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # define dataset
    dataset = ABIDE(argv.sourcedir, roi=argv.roi, dynamic_length=argv.dynamic_length, k_fold=argv.k_fold, smoothing_fwhm=argv.fwhm)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    logger = util.logger.LoggerIMAGIN(argv.k_fold, dataset.num_classes)

    for k in range(argv.k_fold):
        os.makedirs(os.path.join(argv.targetdir, 'attention', str(k)), exist_ok=True)

        model = ModelIMAGIN(
            input_dims=[dataset.aal_num_nodes, dataset.cc200_num_nodes, dataset.schaefer_num_nodes],
            hidden_dims=argv.hidden_dims,
            num_classes=2,
            num_layers=argv.num_layers,
            sparsities=argv.sparsities,
        )
        model.to(device)
        model.load_state_dict(torch.load(os.path.join(argv.targetdir, 'model', str(k), 'model.pth')))
        criterion = torch.nn.CrossEntropyLoss()

        # define logging objects
        fold_attention = {'node_attention': [], 'time_attention': []}
        summary_writer = SummaryWriter(os.path.join(argv.targetdir, 'summary', str(k), 'test'))

        logger.initialize(k)
        dataset.set_fold(k, train=False)
        loss_accumulate = 0.0
        reg_ortho_accumulate = 0.0
        latent_accumulate = []
        for i, x in enumerate(tqdm(dataloader, ncols=60, desc=f'k:{k}')):
            with torch.no_grad():
                # process input data
                dyn_a, sampling_points = util.bold.process_dynamic_fc(x['timeseries'], argv.window_size, argv.window_stride)
                sampling_endpoints = [p+argv.window_size for p in sampling_points]
                if i==0: dyn_v = repeat(torch.eye(dataset.num_nodes), 'n1 n2 -> b t n1 n2', t=len(sampling_points), b=argv.minibatch_size)
                if not dyn_v.shape[1]==dyn_a.shape[1]: dyn_v = repeat(torch.eye(dataset.num_nodes), 'n1 n2 -> b t n1 n2', t=len(sampling_points), b=argv.minibatch_size)
                if len(dyn_a) < argv.minibatch_size: dyn_v = dyn_v[:len(dyn_a)]
                t = x['timeseries'].permute(1,0,2)
                label = x['label']

                logit, loss, attention, latent, reg_ortho = step(
                    model=model,
                    criterion=criterion,
                    dyn_v=dyn_v,
                    dyn_a=dyn_a,
                    sampling_endpoints=sampling_endpoints,
                    t=t,
                    label=label,
                    reg_lambda=argv.reg_lambda,
                    clip_grad=argv.clip_grad,
                    device=device,
                    optimizer=None,
                    scheduler=None)
                pred = logit.argmax(1)
                prob = logit.softmax(1)
                logger.add(k=k, pred=pred.detach().cpu().numpy(), true=label.detach().cpu().numpy(), prob=prob.detach().cpu().numpy())
                loss_accumulate += loss.detach().cpu().numpy()
                reg_ortho_accumulate += reg_ortho.detach().cpu().numpy()

                fold_attention['node_attention'].append(attention['node-attention'].detach().cpu().numpy())
                fold_attention['time_attention'].append(attention['time-attention'].detach().cpu().numpy())
                latent_accumulate.append(latent.detach().cpu().numpy())

        # summarize results
        samples = logger.get(k)
        metrics = logger.evaluate(k)
        summary_writer.add_scalar('loss', loss_accumulate/len(dataloader))
        summary_writer.add_scalar('reg_ortho', reg_ortho_accumulate/len(dataloader))
        summary_writer.add_pr_curve('precision-recall', samples['true'], samples['prob'][:,1])
        [summary_writer.add_scalar(key, value) for key, value in metrics.items() if not key=='fold']
        [summary_writer.add_image(key, make_grid(value[-1].unsqueeze(1), normalize=True, scale_each=True)) for key, value in attention.items()]
        summary_writer.flush()
        print(metrics)

        # finalize fold
        logger.to_csv(argv.targetdir, k)
        if argv.dataset=='rest':
            [np.save(os.path.join(argv.targetdir, 'attention', str(k), f'{key}.npy'), np.concatenate(value)) for key, value in fold_attention.items()]
        elif argv.dataset=='task':
            for key, value in fold_attention.items():
                os.makedirs(os.path.join(argv.targetdir, 'attention', str(k), key), exist_ok=True)
                for idx, task in enumerate(dataset.task_list):
                    np.save(os.path.join(argv.targetdir, 'attention', str(k), key, f'{task}.npy'), np.concatenate([v for (v, l) in zip(value, samples['true']) if l==idx]))
        else:
            raise
        np.save(os.path.join(argv.targetdir, 'attention', str(k), 'latent.npy'), np.concatenate(latent_accumulate))
        del fold_attention

    # finalize experiment
    logger.to_csv(argv.targetdir)
    final_metrics = logger.evaluate()
    print(final_metrics)
    summary_writer.close()
    torch.save(logger.get(), os.path.join(argv.targetdir, 'samples.pkl'))
