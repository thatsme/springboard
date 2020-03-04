

N_months = data_dask.shape[0]
import matplotlib.pyplot as plt
fig, panels = plt.subplots(nrows=4, ncols=3)

for month, panel in zip(range(N_months), panels.flatten()):
    im = panel.imshow(data_dask[month, :, :],
                      origin='lower',
                      vmin=lo, vmax=hi)
    
    panel.set_title('bla bla bla {:02d}'.format(month+1))
    panel.axis('off')
    
plt.suptitle('Monthly averages (max. daily temperature [c])')
plt.colbar(im, ax=panels.ravel().tolist())
plt.show()

