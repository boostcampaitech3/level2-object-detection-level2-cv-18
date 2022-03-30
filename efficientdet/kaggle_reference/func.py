def collate_fn(batch):
    return tuple(zip(*batch))
    

def save(model, optimizer, scheduler, loss, epoch, path):
    model.eval()
    torch.save({
        'model_state_dict': model.model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_summary_loss': loss,
        'epoch': epoch,
    }, path)

    
def log(message, log_path):
    print(message)
    with open(log_path, 'a+') as logger:
        logger.write(f'{message}\n')