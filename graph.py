import matplotlib.pyplot as plt

x = [40, 30, 20, 15, 10, 5]

y_r1 = [0.4331, 0.4423, 0.436, 0.4235, 0.4164, 0.3958]
y_r2 = [0.1744, 0.1728, 0.1741, 0.1683, 0.1586, 0.1398]
y_rl = [0.4027, 0.4137, 0.4077, 0.3957, 0.3878, 0.3702]

graph, (plot1) = plt.subplots(1, 1, figsize=(6, 5))
graph.tight_layout(pad=3.0)
plot1.plot(x, y_r1, color='blue', label='R1')
plot1.plot(x, y_r2, color='orange', label='R2')
plot1.plot(x, y_rl, color='green', label='RL')
plot1.legend()
plot1.set_title("ROUGE comparison")
plot1.set_xlabel('# of sentences')
plot1.set_ylabel('Score')
plot1.invert_xaxis()

plt.savefig("test.png")#, transparent=True)