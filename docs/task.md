## Task format
All data will be in csv format   
No header needed     
Each token have to separate by space    

### once
[example file](https://github.com/voidful/TFkit/blob/master/tfkit/demo_data/generate.csv)   

Format:   
`input, target`    

Example:     
```
"i go to school by bus","我 坐 巴 士 上 學"
```

### onebyone
[example file](https://github.com/voidful/TFkit/blob/master/tfkit/demo_data/generate.csv)   

Format:    
`input, target `    

Example:    
```
"i go to school by bus","我 坐 巴 士 上 學"
```
### qa
[example file](https://github.com/voidful/TFkit/blob/master/tfkit/demo_data/qa.csv)   

Format:    
`input, start_pos, end_pos`      

Example:   
```
"在 歐 洲 , 梵 語 的 學 術 研 究 , 由 德 國 學 者 陸 特 和 漢 斯 雷 頓 開 創 。 後 來 威 廉 · 瓊 斯 發 現 印 歐 語 系 , 也 要 歸 功 於 對 梵 語 的 研 究 。 此 外 , 梵 語 研 究 , 也 對 西 方 文 字 學 及 歷 史 語 言 學 的 發 展 , 貢 獻 不 少 。 1 7 8 6 年 2 月 2 日 , 亞 洲 協 會 在 加 爾 各 答 舉 行 。 會 中 , 威 廉 · 瓊 斯 發 表 了 下 面 這 段 著 名 的 言 論 : 「 梵 語 儘 管 非 常 古 老 , 構 造 卻 精 妙 絕 倫 : 比 希 臘 語 還 完 美 , 比 拉 丁 語 還 豐 富 , 精 緻 之 處 同 時 勝 過 此 兩 者 , 但 在 動 詞 詞 根 和 語 法 形 式 上 , 又 跟 此 兩 者 無 比 相 似 , 不 可 能 是 巧 合 的 結 果 。 這 三 種 語 言 太 相 似 了 , 使 任 何 同 時 稽 考 三 者 的 語 文 學 家 都 不 得 不 相 信 三 者 同 出 一 源 , 出 自 一 種 可 能 已 經 消 逝 的 語 言 。 基 於 相 似 的 原 因 , 儘 管 缺 少 同 樣 有 力 的 證 據 , 我 們 可 以 推 想 哥 德 語 和 凱 爾 特 語 , 雖 然 混 入 了 迥 然 不 同 的 語 彙 , 也 與 梵 語 有 著 相 同 的 起 源 ; 而 古 波 斯 語 可 能 也 是 這 一 語 系 的 子 裔 。 」 [Question] 印 歐 語 系 因 為 哪 一 門 語 言 而 被 發 現 ?",47,49
```

### classify
[example file](https://github.com/voidful/TFkit/blob/master/tfkit/demo_data/classification.csv)   

Format:        
`input,target_0,target_1.....`       
If some task have multiple label, use / to separate each label - label1/label2/label3         

Example:      
```
"The prospective ultrasound findings were correlated with the final diagnoses , laparotomy findings , and pathology findings .",outcome/other,1
```
### tagRow
[example file](https://github.com/voidful/TFkit/blob/master/tfkit/demo_data/tag_row.csv)   

Format:      
`input, target`     

Example:        
```
"在 歐 洲 , 梵 語 的 學 術 研 究 , 由 德 國 學 者 陸 特 和 漢 斯 雷 頓 開 創 。 後 來 威 廉 · 瓊 斯 發 現 印 歐 語 系 , 也 要 歸 功 於 對 梵 語 的 研 究 。 此 外 , 梵 語 研 究 , 也 對 西 方 文 字 學 及 歷 史 語 言 學 的 發 展 , 貢 獻 不 少 。 1 7 8 6 年 2 月 2 日 , 亞 洲 協 會 在 加 爾 各 答 舉 行 。 [SEP] 陸 特 和 漢 斯 雷 頓 開 創 了 哪 一 地 區 對 梵 語 的 學 術 研 究 ?",O A A O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O
```
### tagCol
[example file](https://github.com/voidful/TFkit/blob/master/tfkit/demo_data/tag_col.csv)   

Format:      
`input, target`       

Example:         
```
別 O
只 O
能 R
想 O
自 O
己 O
， O
想 M
你 M
周 O
圍 O
的 O
人 O
。 O
```
