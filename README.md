
## Setup Instructions (Local)

1. Clone the repository:

   ```bash
   git clone https://github.com/kzahiri1/Semiparametric-Regression.git
   cd Semiparametric-Regression
   ```

2. Create environment and install dependencies

   First, create environment using venv or conda and activate your environment.
   Using venv:

   ```bash
   python -m venv <path_to_env>
   source <path_to_env>/bin/activate
   ```

   Using conda:

   ```bash
   conda create -n <env_name> python=3.13.5
   conda activate <env_name>
   ```

   Then, install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Plotly Dash application and open the url shown in terminal ([http://localhost:8050](http://localhost:8050)).

   ```bash
   python main.py
   ```
