import matplotlib
import matplotlib.pyplot as plt
import csv

x = []
results = [[], [], [], []]
with open('numpy_results.csv', 'r') as f:
	for row in csv.reader(f):
		x.append(row[0])
		results[0].append(float(row[4]))
		results[1].append(float(row[3]))
		results[2].append(float(row[2]))
		results[3].append(float(row[1]))
print(results)

width = .2
fig, ax = plt.subplots()
rects1 = ax.bar([index for index, _ in enumerate(x)], results[3], width, color='c')

rects2 = ax.bar([index + width for index, _ in enumerate(x)], results[2], width, color='g')
rects3 = ax.bar([index + width * 2 for index, _ in enumerate(x)], results[1], width, color='b')
rects4 = ax.bar([index + width * 3 for index, _ in enumerate(x)], results[0], width, color='m')

ax.set_title('Speedup over Numpy Convolve Implementation')
ax.set_ylabel('Speedup')
ax.set_xlabel('Square Matrix Size')
ax.set_xticks([index + width for index, _ in enumerate(x)])
ax.set_xticklabels(x)

# ax.legend((rects1[0], rects2[0], rects3[0]), ('Fused v1', 'Unfused', 'Numpy'), loc=2)
ax.legend((rects1[0], rects2[0], rects3[0], rects4[0]), ('SciPy', 'Naive Sequential Specializers (Data Movement)', 'Unfused Kernels (No data movement)','Fused Kernels (No data movement)'), loc=2,
          prop={'size':20})
font = {'family' : 'normal',
                'size'   : 20}

matplotlib.rc('font', **font)
plt.show()
