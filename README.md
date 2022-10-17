# Corrseek

a highly opinionated correlation seeking package.

## backlog

- corr to ensure variables are independent
  - using VIF to detect multicolinearity
- (option) preprocessing to range 0\~1 or -1\~1
- support config parsing and validation
- using log to check for cliffs?
- PELT
- identify date and sort by date
- dropna
- (idea) read csv once with index that make sense, sort along with each bootstrapped model and reset index based on index
- supporting group analysis
  - multicolumn sort
- taichi lang for speedup

```python
df = pd.read_csv("out.csv")
print(id(df))  # >>> 2848833486984
tmp = df.iloc[:, :1]
print(id(tmp))  # >>> 2848833488008
```