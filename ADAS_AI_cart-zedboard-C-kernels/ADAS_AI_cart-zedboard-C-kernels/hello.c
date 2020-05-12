#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/module.h>
#include <linux/fs.h>
#include <asm/uaccess.h>
	
#define FOK 0;
	
volatile static int is_open = 0;
static char msg[1024];
int num_bytes;
static int devnum = 0;

	
	
ssize_t hello_read (struct file * filep, char __user * outb, size_t nby, loff_t * offset )
{
int bytes_read = 0;
if (offset == NULL) return -EINVAL;


if (*offset >= num_bytes) return 0;
while ((bytes_read < nby)&&(*offset < num_bytes)) {
	
put_user (msg[*offset], &outb[bytes_read]);
		
*offset = *offset + 1;
	
bytes_read = bytes_read + 1;

	}

return bytes_read;
	
	
}
ssize_t hello_write (struct file * filep, const char __user * inpb, size_t n_by, loff_t * offset)
{
printk(KERN_ALERT "ERROR - Write Handle Failed\n");
return -EINVAL;
	
}

int hello_open (struct inode * inodep, struct file * filep)
{
	if(is_open == 1)



	{
printk(KERN_INFO "Error - hello device aallready opened!");
		return -EBUSY;
	}
	is_open = 1;
	 try_module_get(THIS_MODULE);
return FOK;
	
}
	
int hello_release (struct inode * inodep, struct file * filep)
{
if(is_open == 0)
	{
		printk(KERN_INFO "ERROR - No instance of hello devices are open! \n");
		return -EBUSY;
	}
	is_open = 0;
	 module_put(THIS_MODULE);
return FOK;






	
}

struct file_operations fops = {
	 read: hello_read,
	 write: hello_write,
	 open: hello_open,
	 release: hello_release
};

static int __init hello_start(void)

{
devnum = register_chrdev(0, "hello", &fops);

	strncpy(msg,"Hello Everyone.",1023);
	 num_bytes = strlen(msg);
	
printk(KERN_INFO "device is :%d\n",devnum);
	printk(KERN_INFO "hello\n");
	return 0;
}

static void __exit hello_end(void)
{
	printk(KERN_INFO "goodbye, see you next time\n");
	
}

module_init(hello_start);
module_exit(hello_end);


