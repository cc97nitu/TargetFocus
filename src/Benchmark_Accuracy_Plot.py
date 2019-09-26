import seaborn as sns
import matplotlib.pyplot as plt

import SQL

if __name__ == "__main__":
    benchResults = SQL.retrieveBenchmark(20)

    accuracyPredictions = benchResults["accuracyPredictions"]

    plot = sns.relplot(x="observed", y="predicted", data=accuracyPredictions)
    plt.plot([-100, 100], [-100, 100], color="orange", scalex=False, scaley=False)

    plt.show()
    plt.close()