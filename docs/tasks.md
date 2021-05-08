## Task format

### Classification

!!! info 
    #### multi-class classification:
    Format: 
    `input sentence,label`     
    
    Example:      
    ```
    Calotropis procera (ushaar) keratitis.,Not-Related
    ```
    
    ####  multi-label classification
    use `///` to separate each label.
    
    Format: 
    `input sentence,label1///label2`  
    
    [Example](https://github.com/voidful/TFkit/blob/master/tfkit/demo_data/classification.csv):      
    ```
    We report two cases of pseudoporphyria caused by naproxen and oxaprozin.,Related///METHODS
    ``` 

### Text Generation

!!! info
    Format:   
    `input sentence, target sentence`
    
    [Example](https://github.com/voidful/TFkit/blob/master/tfkit/demo_data/generation.csv):     
    ```
    Peter was a truck driver . He was running a little behind on schedule . Peter decided to run past the weigh station . He was stopped by a cop .,"Peter ended up running late and getting a fine ."
    ```

### Extractive Question Answering

!!! info
    Format:    
    `input sentence with question, answer start position, answer end position`      
    
    [Example](https://github.com/voidful/TFkit/blob/master/tfkit/demo_data/qa.csv):   
    ```
    Beyoncé announced a hiatus from her music ... <s> Who suggested the hiatus for Beyoncé?, 74,84
    ```

### Multiple-Choice Question Answering

!!! info     
    Input passage should include all available, $each choice must start with a mask token$  
    choice id will be start from 0  
    
    Format:    
    `input passage [MASK]choiceA [MASK]choiceB, 1`      
    
    [Example](https://github.com/voidful/TFkit/blob/master/tfkit/demo_data/mcq.csv):   
    ```
    "I 'm sure many of you have seen Star Wars ... </s> What is the best title of the passage ? [MASK] What Is Human Cloning [MASK] How Does Human Cloning Happen [MASK] Human Cloning Is Wrong [MASK] Discussion On Human Cloning",2
    ```

### Mask Language Modeling

!!! info    
    input sentence with mask, can be multiple     
    target of each mask should be separate by blank     
    Format:    
    `input sentence with [MASK] [MASK],target_token target_token`      
    
    [Example](https://github.com/voidful/TFkit/blob/master/tfkit/demo_data/mask.csv):   
    ```
    "how did i [MASK] [MASK]","get here"
    ```

### Sequence Tagging

!!! info    
    input sentence with blank between each word    
    target label separate with blank, should be one to one to the input    
    Format:     
    `input sentence,tag tag`      
    
    [Example](https://github.com/voidful/TFkit/blob/master/tfkit/demo_data/tag.csv):   
    ```
    "welcome to New York,O O B_place B_place"
    ```
