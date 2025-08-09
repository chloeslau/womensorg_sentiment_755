---
editor: visual
execute:
  echo: true
  output: true
title: Neural Network Predictive Modeling
toc-title: Table of contents
---

::: cell
``` {.r .cell-code}
options(conflicts.policy = "depends.ok")
library(tidyverse)
library(Matrix, exclude = c("expand", "pack", "unpack"))
library(lme4)
library(stringr)
library(broom)
library(skimr)
library(car, exclude = c("recode", "some"))
library(rmemore)
library(ggplot2)
library(lavaan)
library(lme4)
library(lubridate)

theme_set(theme_classic()) 

devtools::source_url("https://github.com/jjcurtin/lab_support/blob/main/fun_plots.R?raw=true")

path_data <- "data"
```
:::

# read in full dataset

::::: cell
``` {.r .cell-code}
df <- read_csv(here::here(path_data, "womens_orgs_WI_FB.csv"), show_col_types = FALSE)
```

::: {.cell-output .cell-output-stderr}
    Warning: One or more parsing issues, call `problems()` on your data frame for details,
    e.g.:
      dat <- vroom(...)
      problems(dat)
:::

``` {.r .cell-code}
names(df)
```

::: {.cell-output .cell-output-stdout}
     [1] "platformId"                               
     [2] "platform"                                 
     [3] "date"                                     
     [4] "updated"                                  
     [5] "type"                                     
     [6] "title"                                    
     [7] "caption"                                  
     [8] "description"                              
     [9] "message"                                  
    [10] "expandedLinks"                            
    [11] "link"                                     
    [12] "postUrl"                                  
    [13] "subscriberCount"                          
    [14] "score"                                    
    [15] "statistics.actual.likeCount"              
    [16] "statistics.actual.shareCount"             
    [17] "statistics.actual.commentCount"           
    [18] "statistics.actual.loveCount"              
    [19] "statistics.actual.wowCount"               
    [20] "statistics.actual.hahaCount"              
    [21] "statistics.actual.sadCount"               
    [22] "statistics.actual.angryCount"             
    [23] "statistics.actual.thankfulCount"          
    [24] "statistics.actual.careCount"              
    [25] "statistics.expected.likeCount"            
    [26] "statistics.expected.shareCount"           
    [27] "statistics.expected.commentCount"         
    [28] "statistics.expected.loveCount"            
    [29] "statistics.expected.wowCount"             
    [30] "statistics.expected.hahaCount"            
    [31] "statistics.expected.sadCount"             
    [32] "statistics.expected.angryCount"           
    [33] "statistics.expected.thankfulCount"        
    [34] "statistics.expected.careCount"            
    [35] "account.id"                               
    [36] "account.name"                             
    [37] "account.handle"                           
    [38] "account.profileImage"                     
    [39] "account.subscriberCount"                  
    [40] "account.url"                              
    [41] "account.platform"                         
    [42] "account.platformId"                       
    [43] "account.accountType"                      
    [44] "account.pageAdminTopCountry"              
    [45] "account.pageDescription"                  
    [46] "account.pageCreatedDate"                  
    [47] "account.pageCategory"                     
    [48] "account.verified"                         
    [49] "languageCode"                             
    [50] "legacyId"                                 
    [51] "id"                                       
    [52] "media"                                    
    [53] "videoLengthMS"                            
    [54] "imageText"                                
    [55] "liveVideoStatus"                          
    [56] "brandedContentSponsor.id"                 
    [57] "brandedContentSponsor.name"               
    [58] "brandedContentSponsor.handle"             
    [59] "brandedContentSponsor.profileImage"       
    [60] "brandedContentSponsor.subscriberCount"    
    [61] "brandedContentSponsor.url"                
    [62] "brandedContentSponsor.platform"           
    [63] "brandedContentSponsor.platformId"         
    [64] "brandedContentSponsor.accountType"        
    [65] "brandedContentSponsor.pageAdminTopCountry"
    [66] "brandedContentSponsor.pageDescription"    
    [67] "brandedContentSponsor.pageCreatedDate"    
    [68] "brandedContentSponsor.pageCategory"       
    [69] "brandedContentSponsor.verified"           
:::
:::::

::::: cell
``` {.r .cell-code}
# unique organizations
df_t <- df |>
  select(account.name, account.handle) |>
  mutate(account.name = as.factor(account.name),
         account.handle = as.factor(account.handle))

df_t |> count(account.handle)
```

::: {.cell-output .cell-output-stdout}
    # A tibble: 16 × 2
       account.handle            n
       <fct>                 <int>
     1 EmergeWisconsin         703
     2 GOTRsewi               3027
     3 LWVWI.ORG              4257
     4 NOWMadison             1261
     5 TEMPOMilwaukee         1732
     6 TheFFBWW               2740
     7 WIWomen                1013
     8 wiwomeninconservation  1561
     9 WIWomensHealth         2093
    10 wiwomensnetwork        2887
    11 WMFWisconsin            517
    12 womensactioncoalition    25
    13 womensfundfvr          1383
    14 womensfundgb           1242
    15 womensfundmke          1840
    16 <NA>                   4227
:::

``` {.r .cell-code}
df_t |> count(account.name)
```

::: {.cell-output .cell-output-stdout}
    # A tibble: 18 × 2
       account.name                                         n
       <fct>                                            <int>
     1 Emerge Wisconsin                                   703
     2 Girls on the Run of Southeastern Wisconsin        3027
     3 League of Women Voters of Wisconsin               4257
     4 Madison NOW - National Organization for Women     1261
     5 TEMPO Milwaukee                                   1732
     6 The Foundation for Black Women's Wellness         2740
     7 Wisconsin African American Women's Center (WAAW)   432
     8 Wisconsin Women's Council                         1013
     9 Wisconsin Women's Health Foundation               2093
    10 Wisconsin Women's Network                         2887
    11 Wisconsin Women in Conservation                   1561
    12 WMF Wisconsin                                      517
    13 Women's Action Coalition                            25
    14 Women's Fund for the Fox Valley Region            1383
    15 Women's Fund of Greater Green Bay                 1242
    16 Women's Fund of Greater Milwaukee                 1840
    17 Women's Health Connection Wisconsin               1685
    18 <NA>                                              2110
:::
:::::

> Given the inconsistent number of levels, we believe that
> "account.name" may be a more accurate "username" type metric.

# Variable selection

::: cell
``` {.r .cell-code}
df1 <- df |>
  select(date, type, 
         title, caption, description, message,
         statistics.actual.likeCount, statistics.actual.shareCount, statistics.actual.commentCount,
         account.name, account.handle, account.subscriberCount, account.pageDescription,
         account.pageCreatedDate, account.pageCategory, account.verified, 
         media)

df1 |> write_csv(here::here(path_data, "cleaned_womens_orgs_WI_FB.csv"))
```
:::

> Process for selecting variables was decided while looking at the raw
> data, checking to see if responses were understandable/comprehensible,
> or could be cleaned. Uploaded for missingness EDA in Python.

# Checking variable levels

Type:

::::: cell
``` {.r .cell-code}
df2 <- df1 |>
  mutate(type = factor(type),
         acc_category = account.pageCategory)
         
# check levels of 'type'
levels(df2$type)
```

::: {.cell-output .cell-output-stdout}
     [1] "link"                   "live_video"             "live_video_complete"   
     [4] "live_video_scheduled"   "native_video"           "photo"                 
     [7] "POLITICAL_ORGANIZATION" "status"                 "video"                 
    [10] "youtube"               
:::

``` {.r .cell-code}
df2 |> count(type)
```

::: {.cell-output .cell-output-stdout}
    # A tibble: 11 × 2
       type                       n
       <fct>                  <int>
     1 link                    9558
     2 live_video                 1
     3 live_video_complete      352
     4 live_video_scheduled     446
     5 native_video            1373
     6 photo                  14385
     7 POLITICAL_ORGANIZATION   703
     8 status                  1898
     9 video                     64
    10 youtube                  321
    11 <NA>                    1407
:::
:::::

> All reasonable types, 1407 NA which we will keep an eye on but is
> about 5% of dataset so can be imputed later.

Account verified:

::::: cell
``` {.r .cell-code}
unique(df2$account.verified)
```

::: {.cell-output .cell-output-stdout}
    [1] "and connections."           "FALSE"                     
    [3] "NON_PROFIT"                 "2010-01-27 05:03:43"       
    [5] NA                           "ENVIRONMENTAL_CONSERVATION"
:::

``` {.r .cell-code}
df2 |> count(account.verified)
```

::: {.cell-output .cell-output-stdout}
    # A tibble: 6 × 2
      account.verified               n
      <chr>                      <int>
    1 2010-01-27 05:03:43         3027
    2 ENVIRONMENTAL_CONSERVATION  1561
    3 FALSE                      18127
    4 NON_PROFIT                  2093
    5 and connections.            2887
    6 <NA>                        2813
:::
:::::

> Date value and "and connections" are confusions here, caused by some
> kind of dataset issue. The date is, from rudimentary research on the
> platform, the date of verification, and the organization is under
> general verification. "and connections" is part of a sentence that got
> clipped across columns and does not particularly indicate verification
> or otherwise, so will be transformed to NA. With that, we end up with
> about 20% missing data. Could still reasonably impute.

:::: cell
``` {.r .cell-code}
df2 <- df2 |>
  mutate(account.verified = if_else(account.verified == "and connections.", NA, account.verified))

df2 <- df2 |>
  mutate(account_verified_binary = if_else(account.verified == "FALSE", "no", "yes"))

# check to make sure it worked
df2 |> count(account_verified_binary)
```

::: {.cell-output .cell-output-stdout}
    # A tibble: 3 × 2
      account_verified_binary     n
      <chr>                   <int>
    1 no                      18127
    2 yes                      6681
    3 <NA>                     5700
:::
::::

Account category

::::: cell
``` {.r .cell-code}
unique(df2$acc_category)
```

::: {.cell-output .cell-output-stdout}
     [1] "advocacy"                          "CHARITY_ORGANIZATION"             
     [3] "NON_PROFIT"                        "2009-09-29 16:07:26"              
     [5] "experience-based curriculum which" "POLITICAL_ORGANIZATION"           
     [7] "LOCAL"                             "GOVERNMENT_ORGANIZATION"          
     [9] NA                                  "2021-01-28 07:43:46"              
    [11] "COMMUNITY_CENTER"                  "COMMUNITY"                        
:::

``` {.r .cell-code}
df2 |> count(acc_category)
```

::: {.cell-output .cell-output-stdout}
    # A tibble: 12 × 2
       acc_category                          n
       <chr>                             <int>
     1 2009-09-29 16:07:26                2093
     2 2021-01-28 07:43:46                1561
     3 CHARITY_ORGANIZATION               2357
     4 COMMUNITY                            25
     5 COMMUNITY_CENTER                   1685
     6 GOVERNMENT_ORGANIZATION            1013
     7 LOCAL                               432
     8 NON_PROFIT                         8358
     9 POLITICAL_ORGANIZATION             4257
    10 advocacy                           2887
    11 experience-based curriculum which  3027
    12 <NA>                               2813
:::
:::::

> Useful variable that we want to keep, especially since multiple
> organizations (account.name) are associated with certain categories,
> but date variables cause some issues. Will bucket into a "None/other"
> category, and impute NAs during recipe building.

:::: cell
``` {.r .cell-code}
# create an "other" bucket
df2 <- df2 |>
  mutate(acc_category = if_else(acc_category == "2009-09-29 16:07:26", "Other", acc_category),
         acc_category = if_else(acc_category == "2021-01-28 07:43:46", "Other", acc_category),
         acc_category = if_else(acc_category == "experience-based curriculum which", 
                                "Other", acc_category))
df2 <- df2 |>
  mutate(acc_category = factor(acc_category))

df2 |> count(acc_category)
```

::: {.cell-output .cell-output-stdout}
    # A tibble: 10 × 2
       acc_category                n
       <fct>                   <int>
     1 advocacy                 2887
     2 CHARITY_ORGANIZATION     2357
     3 COMMUNITY                  25
     4 COMMUNITY_CENTER         1685
     5 GOVERNMENT_ORGANIZATION  1013
     6 LOCAL                     432
     7 NON_PROFIT               8358
     8 Other                    6681
     9 POLITICAL_ORGANIZATION   4257
    10 <NA>                     2813
:::
::::

Media

::: cell
``` {.r .cell-code}
# unique(df2$media)
```
:::

> While a promising variable during eyeballing, cross-referencing
> between Python EDA and this file, the unique levels cannot be cleaned
> appropriately.

# Addressing missingness

## Removing empty observations

::: cell
``` {.r .cell-code}
# type cleaning
df3 <- df2 |>
  filter(!date == "NON_PROFIT") |>
  filter(!date == "and providing a powerful network.") |>
  filter(!date == "and leadership in this country in")

# df3 |> write_csv(here::here(path_data, "2cleaned_womens_orgs_WI_FB.csv"))
```
:::

> Cross-referenced with missingness EDA, rows that do not have a valid
> datetime are empty. This drops 2110 observations. Continuous
> cross-reference with missingness EDA shows that this removes a good
> portion of the missingness we saw in the original EDA. (Completed in
> tandem during class.)

## New binary variables

::: cell
``` {.r .cell-code}
df4 <- df3 |>
  mutate(title_binary = if_else(is.na(title), "no text", "has title"),
         caption_binary = if_else(is.na(caption), "no text", "has caption"),
         description_binary = if_else(is.na(description), "no text", "has desc"))
```
:::

> The next biggest concern was that many posts did not have titles,
> captions, or descriptions. These are optional formats in Facebook
> posts, which makes sense. After looking at many of the unique outputs,
> these strings could be useful with sentiment analysis added, but the
> strong amount of missingness means that imputation may not be
> meaningful. We have opted to create binaries for if posts do or do not
> have titles, captions, and descriptions; these are eye-grabbing parts
> of a post that may determine engagement. Future directions may include
> more nuanced sentiment analysis or NLP.

# Changing date variables

> Tidymodels is capable of handling datetime format data, so this will
> be applied as best practice for this variable format.

# Date split

::: cell
``` {.r .cell-code}
df5 <- df4 |>
  mutate(time_posted = str_extract(date, "(?<= ).*"),
         date_posted = str_extract(date, "^([^ ]+)")) |>
  mutate(time_posted = as.POSIXct(time_posted, format = "%H:%M:%S"),
         date_posted = as.Date(date_posted))
```
:::

:::: cell
``` {.r .cell-code}
df5 |>
  count(account.pageCreatedDate)
```

::: {.cell-output .cell-output-stdout}
    # A tibble: 17 × 2
       account.pageCreatedDate                                                     n
       <chr>                                                                   <int>
     1 2009-04-24 16:46:47                                                      1840
     2 2009-07-15 00:46:53                                                      1261
     3 2010-02-12 14:35:33                                                      1242
     4 2010-08-13 14:30:48                                                      1383
     5 2011-06-02 20:45:12                                                      4257
     6 2011-07-15 16:12:32                                                      1732
     7 2011-09-21 03:39:22                                                       432
     8 2012-02-09 21:20:54                                                      1013
     9 2012-04-20 15:11:18                                                       517
    10 2012-07-28 22:55:15                                                      2740
    11 2019-10-28 21:25:38                                                        25
    12 2022-03-16 01:53:09                                                      1685
    13 education                                                                2887
    14 healthy and confident using a fun                                        3027
    15 improve & impact women's health.                                         2093
    16 landowners and professionals passionately taking good care of soil and…  1561
    17 <NA>                                                                      703
:::

``` {.r .cell-code}
df5 <- df5 |>
  mutate(account.pageCreatedDate = as.POSIXct(account.pageCreatedDate, format = "%Y-%m-%d %H:%M:%S"))
```
::::

> Due to the existence of some non-date variables, which are
> automatically transformed into NAs by the as.POSIXct (datetime format)
> we end up with 10271 NAs. This could plausibly be imputed, but again
> the amount of missingness is very high. This variable will not be
> used. Unlike title, caption, and description, the lack of a creation
> date is not meaningful to how many likes a post gets.

# Save cleaned file

::: cell
``` {.r .cell-code}
df5 |>
  write_csv(here::here(path_data, "sentiment_women_v4.csv"))
```
:::
