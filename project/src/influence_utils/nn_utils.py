import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import ipdb


def inf_iter(dataloader, start_step=0):
    epoch = start_step // len(dataloader)
    while True:
        for batch in dataloader:
            yield epoch, batch

        epoch += 1

            
def compute_gradients_with_loss(
    model,
    loss, 
    optimizer_grouped_parameters=None, 
    create_graph=False
):

    grad = torch.autograd.grad(loss, [p for p in model.parameters()], create_graph=create_graph)
    grad = list(grad)
    
    if optimizer_grouped_parameters:
        for i, (n, p)  in enumerate(model.named_parameters()):
            grad[i] = grad[i] - p * optimizer_grouped_parameters[n]
        
    return grad


def compute_s_test(
    config, 
    model, 
    train_dataset, 
    eval_dataset, 
    data_collator, 
    optimizer_grouped_parameters, 
    epochs=5
):
  
    eval_dataloader = DataLoader(
        eval_dataset, 
        batch_size=config.per_device_batch_size, 
        collate_fn=data_collator
    )
    
    # compute `test_grad_no_reg`
    total_test_loss = []
    test_grad_no_reg = None
    for _ in range(epochs):
        for step, batch in enumerate(eval_dataloader):
            
            model.zero_grad(set_to_none=True)
            
            batch = {k: v.cuda() for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs.loss
            
            #total_test_loss.append(loss.repeat(eval_dataloader.batch_size))
            
            grad = compute_gradients_with_loss(model, loss*len(batch["input_ids"]))
            if test_grad_no_reg is None:
                test_grad_no_reg = grad
            else:
                for i, g in enumerate(grad):
                    test_grad_no_reg[i] += g
                    
    
    for i, _ in enumerate(test_grad_no_reg):
        test_grad_no_reg[i] /= len(eval_dataset)*epochs
        
        
    hessian_dataloader = DataLoader(
        train_dataset,
        batch_size=min(config.hessian_approx.batch_size, len(train_dataset)),
        collate_fn=data_collator,
        shuffle=True,
        pin_memory=True
    )
    
    
    inverse_hvp = None
    with tqdm(total=int(config.hessian_approx.num_samples * config.hessian_approx.recursion_depth)) as pbar:
        for _ in range(config.hessian_approx.num_samples):

            last_estimate = test_grad_no_reg

            r = 0
            for _, batch in inf_iter(hessian_dataloader):

                if r >= config.hessian_approx.recursion_depth: break

                model.zero_grad(set_to_none=True)

                batch = {k: v.cuda() for k, v in batch.items()}

                
                outputs = model(**batch)
                loss = outputs.loss
                
                
                first_order_grad = compute_gradients_with_loss(
                    model,
                    loss,  
                    optimizer_grouped_parameters=optimizer_grouped_parameters, 
                    create_graph=True
                )
                
                model.zero_grad(set_to_none=True)

                hessian_vector_val = torch.autograd.grad(
                    first_order_grad, 
                    [p for p in model.parameters()], 
                    grad_outputs=last_estimate, 
                    only_inputs=True
                )


                with torch.no_grad():
                    cur_estimate = [a + (1 - config.hessian_approx.damping) * b - c / config.hessian_approx.scale \
                        for (a,b,c) in zip(test_grad_no_reg, last_estimate, hessian_vector_val)]


                if r % 100 == 0:
                    cur_estimate_norm = cur_estimate[0].norm().item()
                    last_estimate_norm = last_estimate[0].norm().item()
                    norm_diff = cur_estimate_norm - last_estimate_norm
                    pbar.set_description(f'{cur_estimate_norm:.2f} | {norm_diff:.2f}')

                last_estimate = cur_estimate
                pbar.update(1)
                r += 1


            if inverse_hvp is None:
                inverse_hvp = [b / config.hessian_approx.scale for b in cur_estimate]
            else:
                inverse_hvp = [a + b / config.hessian_approx.scale for (a, b) in zip(inverse_hvp, cur_estimate)]

    inverse_hvp = [a / config.hessian_approx.num_samples for a in inverse_hvp]
    return inverse_hvp
    

def compute_influences(
    config, 
    model, 
    train_dataset, 
    eval_dataset, 
    data_collator, 
    optimizer_grouped_parameters
):
    
    model.eval()
    model.cuda()
    
    
    s_test = compute_s_test(
        config, 
        model, 
        train_dataset, 
        eval_dataset, 
        data_collator, 
        optimizer_grouped_parameters
    )
    
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.per_device_batch_size,
        collate_fn=data_collator,
    )
    
    influences = 0.
    
    for batch in train_dataloader:
        model.zero_grad(set_to_none=True)
        
        batch = {k: v.cuda() for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        
        train_grad = compute_gradients_with_loss(model, loss*len(batch["input_ids"]), optimizer_grouped_parameters=optimizer_grouped_parameters)
        influences += sum([(g * ihvp).sum() for g, ihvp in zip(train_grad, s_test)]) / len(train_dataset)
    
    return influences
