# Corrseek

a highly opinionated correlation seeking package.

## proposal

### approach 1

1. training phase - initial phase to learn pairwise correlation (linear's a/b or quadratic's a/b/c) and save to database
2. training phase - define boundaries through 3 sigma outlier
3. inference phase - fill up empties

## backlog

- [ ] corr to ensure variables are independent
  - using VIF to detect multicolinearity
- [ ] (option) preprocessing to range 0\~1 or -1\~1
- [ ] support config parsing and validation
- [ ] using log to check for cliffs?
- [ ] PELT
- [ ] identify date and sort by date
- [ ] dropna
- [ ] (idea) read csv once with index that make sense, sort along with each bootstrapped model and reset index based on index
- [ ] supporting group analysis
  - multicolumn sort
- [ ] taichi lang/jit compiler for speedup
- [ ] proper logging
- assumptions check

```python
df = pd.read_csv("out.csv")
print(id(df))  # >>> 2848833486984
tmp = df.iloc[:, :1]
print(id(tmp))  # >>> 2848833488008
```