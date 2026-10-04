import pandas as pd, glob, os, sys
sys.stdout.reconfigure(encoding="utf-8")
rows=[]
for f in glob.glob("parts_iter/res_*.csv"):
    a=pd.read_csv(f); b=pd.read_csv(os.path.join("parts",os.path.basename(f)))
    k=["업종","h"] if "업종" in a.columns else ["h"]
    m=a.merge(b,on=[c for c in ["업종","h"] if c in a.columns],suffixes=("_i","_f"))
    rows.append({"파일":os.path.basename(f),"칸":len(m),"β3_최대차":(m["β3_i"]-m["β3_f"]).abs().max(),
                 "SE_최대상대차":(m["SE_f"]/m["SE_i"]-1).abs().max(),"N같음":(m["N_i"]==m["N_f"]).all()})
for f in glob.glob("parts_iter/pre_*.csv"):
    a=pd.read_csv(f); b=pd.read_csv(os.path.join("parts",os.path.basename(f)))
    rows.append({"파일":os.path.basename(f),"칸":1,"β3_최대차":abs(a["원본방식_β3pre"][0]-b["원본방식_β3pre"][0]),
                 "SE_최대상대차":abs(b["p_chi2"][0]-a["p_chi2"][0]),"N같음":True})
t=pd.DataFrame(rows); print(t.to_string(index=False)); t.to_csv("fwl_vs_iter_check.csv",index=False,encoding="utf-8-sig")
print("최대 β3 차",t["β3_최대차"].max(),"최대 SE 상대차",t["SE_최대상대차"].max())
