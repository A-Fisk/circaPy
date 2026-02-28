Usage
=====


Installation
------------

Install circaPy from pip:

.. code-block:: console

    pip install circapy

Or install the latest development version directly from GitHub:

.. code-block:: console

    pip install git+https://github.com/A-Fisk/circaPy.git@main


Quickstart
----------

Create sample time-series activity data:

.. code-block:: python

    import pandas as pd
    import numpy as np

    # Create a sample dataset with time-series activity data
    index = pd.date_range(start='2024-01-01', periods=86400, freq="10s")
    values = np.random.randint(0, 100, size=(len(index), 2))

    df = pd.DataFrame(values, index=index)


Calculate Interdaily Variability (IV)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import circaPy.activity as act

    iv = act.calculate_IV(df)
    print(f"Interdaily Variability (IV). Col 0: {iv[0]:.4f}")


Plot an actogram
~~~~~~~~~~~~~~~~~

.. code-block:: python

    import circaPy.plots as cpp

    cpp.plot_actogram(df, showfig=True)


Calculate Intradaily Stability (IS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    is_value = act.calculate_IS(df)
    print(f"Intradaily Stability (IS). Col 0: {is_value[0]:.4f}")
