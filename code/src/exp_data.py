# %%
from AutoCute.code.src.auto_config import project_dir

csv_en_path = project_dir / "temp/test_en.csv"
import pandas as pd
from ydata_profiling import ProfileReport

df = pd.read_csv(csv_en_path)
df
# %%
profile = ProfileReport(df, title="Profiling Report", explorative=True)
profile.to_file(project_dir / "temp/profile_report.html")
# %%
