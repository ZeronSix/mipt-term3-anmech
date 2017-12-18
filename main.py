import numpy as np
import matplotlib.pyplot as plt
import simulator

EXPERIMENT_COUNT = 50

if __name__ == "__main__":
    print("Введите радиус планеты:")
    radius = float(input())
    print("Введите максимальный прицельный параметр:")
    max_impact = float(input())
    print("Введите скорость на бесконечности в долях от первой космической:")
    vel = float(input())
    print("Введите функцию распределения плотности:")
    func = input()
    sim = simulator.Simulator(radius, func, vel)

    impacts = np.linspace(0, max_impact, EXPERIMENT_COUNT, endpoint=True)
    data = []
    for impact in impacts:
        data.append(sim.run(impact))

    min_dists = [x[simulator.INDEX_MIN_DIST] for x in data]
    angles = [x[simulator.INDEX_ANGLE] for x in data]

    optimum = max(data, key=lambda t: t[simulator.INDEX_ANGLE])

    plt.figure(1)
    plt.plot(impacts, min_dists)
    plt.xlabel("Прицельный параметр, R")
    plt.ylabel("Расстояние до перицентра, м")
    plt.title("Зависимость расстояния до перицентра от прицельного параметра")

    plt.figure(2)
    plt.plot(impacts, angles)
    plt.xlabel("Прицельный параметр, R")
    plt.ylabel("Угол отклонения, рад")
    plt.title("Зависимость угла отклонения скорости от прицельного параметра")

    plt.figure(3)
    plt.plot(min_dists, angles)
    plt.xlabel("Расстояние до перицентра, м")
    plt.ylabel("Угол отклонения, рад")
    plt.title("Зависимость угла отклонения скорости от расстояния до перицентра")

    plt.figure(4)
    circle = plt.Circle((0, 0), radius=radius, fill=False)
    ax = plt.gca()
    ax.add_patch(circle)
    plt.xlabel("x, м")
    plt.ylabel("y, м")
    plt.title("Оптимальная траектория")
    plt.plot(*np.transpose(optimum[simulator.INDEX_TRAJECTORY]))

    plt.figure(5)
    plt.xlabel("Время, с")
    plt.ylabel("Полная энергия, Дж/кг")
    plt.title("Зависимость полной энергии от времени для оптимальной траектории")
    plt.plot(*np.transpose(optimum[simulator.INDEX_FULL_ENERGY]))
    plt.show()
