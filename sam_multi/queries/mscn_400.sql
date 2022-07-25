SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND title.kind_id=7 AND movie_info_idx.info_type_id>99
SELECT COUNT(*) FROM title WHERE title.production_year>2004
SELECT COUNT(*) FROM movie_info WHERE movie_info.info_type_id<4
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND movie_companies.company_id<27
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id<55
SELECT COUNT(*) FROM movie_companies,movie_info,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info.movie_id AND title.production_year>1977 AND movie_companies.company_id>71403 AND movie_info.info_type_id<4
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id<35049
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id
SELECT COUNT(*) FROM movie_info_idx WHERE movie_info_idx.info_type_id>99
SELECT COUNT(*) FROM movie_info,movie_info_idx,title WHERE title.id=movie_info.movie_id AND title.id=movie_info_idx.movie_id AND movie_info.info_type_id>16 AND movie_info_idx.info_type_id=100
SELECT COUNT(*) FROM cast_info,movie_keyword,title WHERE title.id=cast_info.movie_id AND title.id=movie_keyword.movie_id AND cast_info.person_id=890821 AND cast_info.role_id=1 AND movie_keyword.keyword_id>3624
SELECT COUNT(*) FROM movie_info,movie_keyword,title WHERE title.id=movie_info.movie_id AND title.id=movie_keyword.movie_id AND title.production_year>2010 AND movie_info.info_type_id<16
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND title.production_year<2010 AND movie_info_idx.info_type_id=101
SELECT COUNT(*) FROM cast_info WHERE cast_info.role_id=10
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.kind_id>1
SELECT COUNT(*) FROM movie_info WHERE movie_info.info_type_id=3
SELECT COUNT(*) FROM movie_info,movie_info_idx,title WHERE title.id=movie_info.movie_id AND title.id=movie_info_idx.movie_id AND title.production_year>2003 AND movie_info.info_type_id=16
SELECT COUNT(*) FROM movie_info WHERE movie_info.info_type_id>16
SELECT COUNT(*) FROM movie_info WHERE movie_info.info_type_id<16
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id=18559
SELECT COUNT(*) FROM cast_info,movie_info_idx,title WHERE title.id=cast_info.movie_id AND title.id=movie_info_idx.movie_id
SELECT COUNT(*) FROM movie_companies,movie_info,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info.movie_id AND movie_info.info_type_id<4
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.production_year<2007 AND movie_keyword.keyword_id>60992
SELECT COUNT(*) FROM cast_info,movie_info,title WHERE title.id=cast_info.movie_id AND title.id=movie_info.movie_id AND title.kind_id=7 AND title.production_year>1998 AND movie_info.info_type_id<15
SELECT COUNT(*) FROM cast_info,movie_info_idx,title WHERE title.id=cast_info.movie_id AND title.id=movie_info_idx.movie_id AND cast_info.person_id>761742 AND cast_info.role_id=3
SELECT COUNT(*) FROM movie_info_idx WHERE movie_info_idx.info_type_id<101
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND title.kind_id>1 AND title.production_year<2010 AND movie_companies.company_type_id<2
SELECT COUNT(*) FROM movie_info,movie_keyword,title WHERE title.id=movie_info.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id<7 AND title.production_year>2000
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND movie_keyword.keyword_id=20450
SELECT COUNT(*) FROM movie_info WHERE movie_info.info_type_id<98
SELECT COUNT(*) FROM movie_companies,movie_info_idx,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info_idx.movie_id AND movie_companies.company_type_id=2
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND movie_info.info_type_id>98
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.production_year<1999
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.kind_id=7 AND title.production_year=1992 AND cast_info.person_id>2415257 AND cast_info.role_id<3
SELECT COUNT(*) FROM movie_info_idx WHERE movie_info_idx.info_type_id=101
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.kind_id=1 AND title.production_year=1993 AND movie_info.info_type_id>3
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_type_id<2
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND movie_keyword.keyword_id>11868
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id>312 AND movie_companies.company_type_id=2
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id>5071
SELECT COUNT(*) FROM movie_info_idx WHERE movie_info_idx.info_type_id<100
SELECT COUNT(*) FROM movie_info_idx WHERE movie_info_idx.info_type_id=100
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.production_year<2001
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id>1423339
SELECT COUNT(*) FROM movie_info,movie_keyword,title WHERE title.id=movie_info.movie_id AND title.id=movie_keyword.movie_id
SELECT COUNT(*) FROM movie_info WHERE movie_info.info_type_id=7
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id<3639
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.production_year=1913 AND movie_info.info_type_id=7
SELECT COUNT(*) FROM title WHERE title.production_year=1993
SELECT COUNT(*) FROM title WHERE title.kind_id>3
SELECT COUNT(*) FROM cast_info WHERE cast_info.role_id>2
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND movie_info.info_type_id<5
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.kind_id>4 AND movie_keyword.keyword_id=5889
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND title.kind_id=7
SELECT COUNT(*) FROM cast_info WHERE cast_info.role_id=3
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id<1700496
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND title.kind_id>1 AND title.production_year<2002
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.kind_id=7 AND title.production_year>2011
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND title.kind_id<4 AND title.production_year>2001
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id>2758
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id>430
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id<15518
SELECT COUNT(*) FROM title WHERE title.production_year<2012
SELECT COUNT(*) FROM title WHERE title.kind_id=7 AND title.production_year>2011
SELECT COUNT(*) FROM movie_info,movie_keyword,title WHERE title.id=movie_info.movie_id AND title.id=movie_keyword.movie_id AND movie_keyword.keyword_id<6258
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id AND title.production_year<2011 AND movie_companies.company_id>1700 AND movie_keyword.keyword_id<746
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND movie_info_idx.info_type_id>101
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id AND title.production_year<2005 AND movie_keyword.keyword_id>807
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id<95839
SELECT COUNT(*) FROM title WHERE title.kind_id=1
SELECT COUNT(*) FROM cast_info,movie_keyword,title WHERE title.id=cast_info.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id<7 AND title.production_year=2003 AND cast_info.person_id<3759296 AND cast_info.role_id=10
SELECT COUNT(*) FROM cast_info WHERE cast_info.role_id=1
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id<865
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id=73864
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.kind_id<7 AND title.production_year<2009 AND cast_info.person_id<1384789 AND cast_info.role_id>1
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.production_year=2008
SELECT COUNT(*) FROM cast_info,movie_info,title WHERE title.id=cast_info.movie_id AND title.id=movie_info.movie_id AND title.production_year<1981 AND cast_info.person_id>3447372 AND cast_info.role_id<10 AND movie_info.info_type_id>3
SELECT COUNT(*) FROM cast_info,movie_info_idx,title WHERE title.id=cast_info.movie_id AND title.id=movie_info_idx.movie_id AND cast_info.role_id>3
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.production_year<2002
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.production_year>2002
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id<21 AND movie_companies.company_type_id<2
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id=7 AND title.production_year<2013 AND movie_companies.company_type_id=1
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.production_year>2010
SELECT COUNT(*) FROM movie_info WHERE movie_info.info_type_id=4
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.kind_id>1 AND title.production_year=2011
SELECT COUNT(*) FROM title WHERE title.production_year>2010
SELECT COUNT(*) FROM movie_info,movie_keyword,title WHERE title.id=movie_info.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id=7 AND title.production_year<1983 AND movie_info.info_type_id=18
SELECT COUNT(*) FROM cast_info,movie_companies,title WHERE title.id=movie_companies.movie_id AND title.id=cast_info.movie_id AND title.production_year>1958
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND title.production_year<2011 AND movie_info_idx.info_type_id<101
SELECT COUNT(*) FROM movie_companies,movie_info,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info.movie_id AND title.kind_id=7 AND movie_companies.company_type_id>1
SELECT COUNT(*) FROM title WHERE title.production_year<1999
SELECT COUNT(*) FROM movie_companies,movie_info_idx,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info_idx.movie_id AND movie_companies.company_id=133 AND movie_companies.company_type_id=1
SELECT COUNT(*) FROM cast_info,movie_keyword,title WHERE title.id=cast_info.movie_id AND title.id=movie_keyword.movie_id AND cast_info.person_id>3028889 AND movie_keyword.keyword_id=6079
SELECT COUNT(*) FROM movie_info WHERE movie_info.info_type_id<18
SELECT COUNT(*) FROM movie_info,movie_info_idx,title WHERE title.id=movie_info.movie_id AND title.id=movie_info_idx.movie_id AND movie_info.info_type_id=3
SELECT COUNT(*) FROM movie_info WHERE movie_info.info_type_id=5
SELECT COUNT(*) FROM movie_companies,movie_info_idx,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id<7 AND title.production_year<1995 AND movie_companies.company_id>93251 AND movie_companies.company_type_id=2
SELECT COUNT(*) FROM cast_info,movie_keyword,title WHERE title.id=cast_info.movie_id AND title.id=movie_keyword.movie_id AND cast_info.role_id>5 AND movie_keyword.keyword_id>21497
SELECT COUNT(*) FROM movie_info,movie_keyword,title WHERE title.id=movie_info.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id=7 AND title.production_year>2009 AND movie_info.info_type_id<73 AND movie_keyword.keyword_id>5346
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id
SELECT COUNT(*) FROM cast_info,movie_info,title WHERE title.id=cast_info.movie_id AND title.id=movie_info.movie_id AND title.kind_id=7 AND cast_info.person_id<2164700 AND cast_info.role_id<10 AND movie_info.info_type_id<17
SELECT COUNT(*) FROM movie_info WHERE movie_info.info_type_id<3
SELECT COUNT(*) FROM movie_info WHERE movie_info.info_type_id>1
SELECT COUNT(*) FROM movie_info,movie_keyword,title WHERE title.id=movie_info.movie_id AND title.id=movie_keyword.movie_id AND movie_info.info_type_id=2 AND movie_keyword.keyword_id>11171
SELECT COUNT(*) FROM title WHERE title.kind_id>4
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.production_year<1928 AND movie_keyword.keyword_id=980
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.kind_id<7
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND movie_companies.company_id>74659
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND movie_info.info_type_id<2
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.production_year<1993 AND cast_info.person_id<1128988 AND cast_info.role_id>10
SELECT COUNT(*) FROM movie_info,movie_keyword,title WHERE title.id=movie_info.movie_id AND title.id=movie_keyword.movie_id AND title.production_year=2005 AND movie_info.info_type_id>18 AND movie_keyword.keyword_id<56
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.kind_id=1 AND title.production_year<1997 AND movie_keyword.keyword_id>24852
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id=3532427
SELECT COUNT(*) FROM cast_info,movie_keyword,title WHERE title.id=cast_info.movie_id AND title.id=movie_keyword.movie_id
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id=1 AND title.production_year<2012 AND movie_keyword.keyword_id<2909
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.production_year>2011 AND movie_keyword.keyword_id<2021
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.production_year<2010
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id<1018786 AND cast_info.role_id=8
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.production_year<2002 AND movie_info.info_type_id<16
SELECT COUNT(*) FROM title WHERE title.production_year>1960
SELECT COUNT(*) FROM cast_info,movie_info_idx,title WHERE title.id=cast_info.movie_id AND title.id=movie_info_idx.movie_id AND movie_info_idx.info_type_id<101
SELECT COUNT(*) FROM movie_info_idx,movie_keyword,title WHERE title.id=movie_info_idx.movie_id AND title.id=movie_keyword.movie_id AND title.production_year=1985 AND movie_info_idx.info_type_id<101
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND movie_keyword.keyword_id=17222
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id<5273
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id
SELECT COUNT(*) FROM movie_info_idx,movie_keyword,title WHERE title.id=movie_info_idx.movie_id AND title.id=movie_keyword.movie_id AND title.production_year>2001 AND movie_info_idx.info_type_id=100
SELECT COUNT(*) FROM movie_info_idx,movie_keyword,title WHERE title.id=movie_info_idx.movie_id AND title.id=movie_keyword.movie_id AND title.production_year=2012 AND movie_keyword.keyword_id>4841
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND cast_info.person_id=518069 AND cast_info.role_id=1
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id<7 AND movie_companies.company_id>11156 AND movie_keyword.keyword_id=1190
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND movie_keyword.keyword_id>1767
SELECT COUNT(*) FROM title WHERE title.kind_id=1 AND title.production_year=2007
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.kind_id<4 AND title.production_year=2004 AND movie_keyword.keyword_id<1633
SELECT COUNT(*) FROM movie_info WHERE movie_info.info_type_id=18
SELECT COUNT(*) FROM movie_info_idx,movie_keyword,title WHERE title.id=movie_info_idx.movie_id AND title.id=movie_keyword.movie_id AND movie_info_idx.info_type_id<100
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.kind_id<4 AND title.production_year<2003 AND movie_keyword.keyword_id>21949
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id AND title.production_year>2012
SELECT COUNT(*) FROM movie_info,movie_info_idx,title WHERE title.id=movie_info.movie_id AND title.id=movie_info_idx.movie_id AND title.production_year>2001 AND movie_info.info_type_id=15
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id>22322
SELECT COUNT(*) FROM cast_info,movie_keyword,title WHERE title.id=cast_info.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id<7 AND movie_keyword.keyword_id<1046
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.kind_id<7 AND title.production_year>2003
SELECT COUNT(*) FROM movie_info WHERE movie_info.info_type_id<15
SELECT COUNT(*) FROM movie_companies,movie_info,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info.movie_id AND movie_companies.company_id<176 AND movie_companies.company_type_id<2
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND movie_keyword.keyword_id=4333
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND title.production_year<1971
SELECT COUNT(*) FROM movie_info,movie_keyword,title WHERE title.id=movie_info.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id<7
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_type_id=1
SELECT COUNT(*) FROM movie_info_idx,movie_keyword,title WHERE title.id=movie_info_idx.movie_id AND title.id=movie_keyword.movie_id AND movie_keyword.keyword_id>394
SELECT COUNT(*) FROM movie_info WHERE movie_info.info_type_id>18
SELECT COUNT(*) FROM movie_info_idx,movie_keyword,title WHERE title.id=movie_info_idx.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id<4
SELECT COUNT(*) FROM cast_info,movie_info_idx,title WHERE title.id=cast_info.movie_id AND title.id=movie_info_idx.movie_id AND cast_info.role_id>1 AND movie_info_idx.info_type_id=99
SELECT COUNT(*) FROM cast_info,movie_keyword,title WHERE title.id=cast_info.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id>4 AND cast_info.role_id<4
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.kind_id>2 AND movie_info.info_type_id>3
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id=1002
SELECT COUNT(*) FROM title WHERE title.kind_id>1 AND title.production_year=1958
SELECT COUNT(*) FROM cast_info,movie_info_idx,title WHERE title.id=cast_info.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id=7 AND title.production_year<1943 AND movie_info_idx.info_type_id=101
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id=71881
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.kind_id=1 AND title.production_year=2012
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id=4 AND title.production_year<2006
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND title.production_year>1986
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id>79391
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id<562
SELECT COUNT(*) FROM cast_info,movie_info,title WHERE title.id=cast_info.movie_id AND title.id=movie_info.movie_id AND title.kind_id=7 AND cast_info.role_id=2
SELECT COUNT(*) FROM title WHERE title.kind_id=7 AND title.production_year>2007
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id=103168
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND title.kind_id>1
SELECT COUNT(*) FROM movie_info_idx,movie_keyword,title WHERE title.id=movie_info_idx.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id=7 AND title.production_year>1976 AND movie_info_idx.info_type_id=99
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.kind_id=1 AND cast_info.role_id>1
SELECT COUNT(*) FROM title WHERE title.kind_id=4 AND title.production_year=2012
SELECT COUNT(*) FROM cast_info,movie_companies,title WHERE title.id=movie_companies.movie_id AND title.id=cast_info.movie_id AND title.kind_id=1 AND title.production_year>2008
SELECT COUNT(*) FROM movie_info WHERE movie_info.info_type_id=105
SELECT COUNT(*) FROM movie_info_idx WHERE movie_info_idx.info_type_id=99
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id AND title.production_year<2011 AND movie_companies.company_id=12929 AND movie_companies.company_type_id<2 AND movie_keyword.keyword_id<3905
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id
SELECT COUNT(*) FROM cast_info,movie_keyword,title WHERE title.id=cast_info.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id<7 AND title.production_year>2012 AND cast_info.role_id>2 AND movie_keyword.keyword_id<394
SELECT COUNT(*) FROM cast_info,movie_info,title WHERE title.id=cast_info.movie_id AND title.id=movie_info.movie_id AND title.kind_id=1 AND cast_info.role_id>2 AND movie_info.info_type_id>8
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id=642522 AND cast_info.role_id<8
SELECT COUNT(*) FROM cast_info,movie_info,title WHERE title.id=cast_info.movie_id AND title.id=movie_info.movie_id AND title.production_year=2006 AND cast_info.role_id>2 AND movie_info.info_type_id<98
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id=15488
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND movie_info_idx.info_type_id<100
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.production_year<2012 AND movie_info.info_type_id=1
SELECT COUNT(*) FROM cast_info,movie_info_idx,title WHERE title.id=cast_info.movie_id AND title.id=movie_info_idx.movie_id AND cast_info.person_id>95397 AND movie_info_idx.info_type_id=101
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.kind_id>1 AND title.production_year<2004
SELECT COUNT(*) FROM title WHERE title.kind_id<7 AND title.production_year<1984
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND title.kind_id=4 AND title.production_year=1999
SELECT COUNT(*) FROM title WHERE title.kind_id=7 AND title.production_year>1913
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id>69766 AND movie_companies.company_type_id<2
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id>3064628
SELECT COUNT(*) FROM movie_info,movie_info_idx,title WHERE title.id=movie_info.movie_id AND title.id=movie_info_idx.movie_id AND movie_info.info_type_id>3
SELECT COUNT(*) FROM movie_companies,movie_info,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info.movie_id AND movie_companies.company_id>11586 AND movie_companies.company_type_id<2
SELECT COUNT(*) FROM cast_info,movie_companies,title WHERE title.id=movie_companies.movie_id AND title.id=cast_info.movie_id AND cast_info.person_id<584885 AND cast_info.role_id<3
SELECT COUNT(*) FROM cast_info,movie_info,title WHERE title.id=cast_info.movie_id AND title.id=movie_info.movie_id AND title.production_year<1977 AND cast_info.role_id=1
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.kind_id<4 AND title.production_year<2012
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.kind_id<7 AND title.production_year>2009 AND movie_keyword.keyword_id>2130
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.production_year>2006 AND movie_info.info_type_id<16
SELECT COUNT(*) FROM movie_info_idx,movie_keyword,title WHERE title.id=movie_info_idx.movie_id AND title.id=movie_keyword.movie_id AND title.production_year>1981 AND movie_keyword.keyword_id>819
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id AND movie_companies.company_id<892 AND movie_keyword.keyword_id<52673
SELECT COUNT(*) FROM title WHERE title.kind_id=2
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND title.kind_id=7 AND movie_info_idx.info_type_id=100
SELECT COUNT(*) FROM title WHERE title.kind_id=1 AND title.production_year=2009
SELECT COUNT(*) FROM title WHERE title.kind_id=7 AND title.production_year>1988
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND title.kind_id=3 AND movie_info_idx.info_type_id<101
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND title.kind_id>1 AND title.production_year<1986 AND movie_companies.company_id<13251 AND movie_companies.company_type_id<2
SELECT COUNT(*) FROM movie_info_idx WHERE movie_info_idx.info_type_id>101
SELECT COUNT(*) FROM cast_info,movie_info_idx,title WHERE title.id=cast_info.movie_id AND title.id=movie_info_idx.movie_id AND title.production_year<2013 AND cast_info.role_id=1 AND movie_info_idx.info_type_id=100
SELECT COUNT(*) FROM cast_info,movie_keyword,title WHERE title.id=cast_info.movie_id AND title.id=movie_keyword.movie_id AND title.production_year=2006
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND movie_companies.company_type_id<2
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id=2435219 AND cast_info.role_id>1
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND title.kind_id=7 AND title.production_year<2005
SELECT COUNT(*) FROM movie_info,movie_info_idx,title WHERE title.id=movie_info.movie_id AND title.id=movie_info_idx.movie_id AND movie_info_idx.info_type_id=99
SELECT COUNT(*) FROM cast_info,movie_keyword,title WHERE title.id=cast_info.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id<7 AND title.production_year<2009 AND cast_info.role_id<11
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id AND movie_companies.company_id>13331 AND movie_keyword.keyword_id=115
SELECT COUNT(*) FROM cast_info,movie_companies,title WHERE title.id=movie_companies.movie_id AND title.id=cast_info.movie_id AND title.kind_id=7 AND cast_info.person_id>2802940 AND cast_info.role_id=9
SELECT COUNT(*) FROM title WHERE title.production_year=2004
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.kind_id=7
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND title.production_year>2006 AND movie_companies.company_id<190835 AND movie_companies.company_type_id=1
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND movie_info.info_type_id>16
SELECT COUNT(*) FROM cast_info WHERE cast_info.role_id=4
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id<1404176 AND cast_info.role_id>3
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.production_year<2004
SELECT COUNT(*) FROM cast_info WHERE cast_info.role_id<11
SELECT COUNT(*) FROM movie_info,movie_info_idx,title WHERE title.id=movie_info.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id=7 AND title.production_year=2008 AND movie_info.info_type_id=18
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.kind_id=7 AND title.production_year<1995
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.kind_id<7 AND title.production_year=1995 AND movie_info.info_type_id>1
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.kind_id=1
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND movie_info_idx.info_type_id=99
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.kind_id=7 AND title.production_year<2008
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id>74318
SELECT COUNT(*) FROM cast_info,movie_keyword,title WHERE title.id=cast_info.movie_id AND title.id=movie_keyword.movie_id AND title.production_year<1984 AND cast_info.role_id=2
SELECT COUNT(*) FROM title WHERE title.production_year=1985
SELECT COUNT(*) FROM cast_info,movie_info_idx,title WHERE title.id=cast_info.movie_id AND title.id=movie_info_idx.movie_id AND cast_info.person_id=3856403 AND cast_info.role_id>2 AND movie_info_idx.info_type_id=101
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id>4834
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id>7634
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.production_year=1998 AND movie_info.info_type_id=7
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND movie_keyword.keyword_id>5603
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.kind_id<7 AND title.production_year=2008
SELECT COUNT(*) FROM movie_info,movie_keyword,title WHERE title.id=movie_info.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id<4 AND title.production_year>2008 AND movie_keyword.keyword_id=3509
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.kind_id<7 AND title.production_year=1950 AND cast_info.person_id>2550841
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id=14698
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND cast_info.role_id<3
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND title.production_year=2013
SELECT COUNT(*) FROM movie_info_idx,movie_keyword,title WHERE title.id=movie_info_idx.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id=4
SELECT COUNT(*) FROM cast_info,movie_info,title WHERE title.id=cast_info.movie_id AND title.id=movie_info.movie_id AND title.production_year=1992 AND movie_info.info_type_id>9
SELECT COUNT(*) FROM movie_info,movie_keyword,title WHERE title.id=movie_info.movie_id AND title.id=movie_keyword.movie_id AND movie_info.info_type_id>16
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id>61606
SELECT COUNT(*) FROM movie_info_idx,movie_keyword,title WHERE title.id=movie_info_idx.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id=7 AND title.production_year>2006 AND movie_keyword.keyword_id>7316
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.kind_id<7 AND title.production_year<1972 AND cast_info.person_id>1463411
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id<453826
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id<3257186 AND cast_info.role_id=1
SELECT COUNT(*) FROM movie_info_idx,movie_keyword,title WHERE title.id=movie_info_idx.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id<7 AND title.production_year<1994
SELECT COUNT(*) FROM title WHERE title.kind_id>3 AND title.production_year>1985
SELECT COUNT(*) FROM movie_info WHERE movie_info.info_type_id<47
SELECT COUNT(*) FROM cast_info,movie_keyword,title WHERE title.id=cast_info.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id<7 AND movie_keyword.keyword_id<3660
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id AND movie_companies.company_id>11137 AND movie_companies.company_type_id=2
SELECT COUNT(*) FROM movie_companies,movie_info_idx,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id<7 AND title.production_year=2005 AND movie_companies.company_type_id=1
SELECT COUNT(*) FROM title WHERE title.production_year=1992
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND cast_info.person_id=318300 AND cast_info.role_id<9
SELECT COUNT(*) FROM title WHERE title.kind_id<7 AND title.production_year<2010
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.kind_id<7 AND cast_info.role_id>10
SELECT COUNT(*) FROM title WHERE title.kind_id=7
SELECT COUNT(*) FROM title WHERE title.kind_id<7 AND title.production_year>2004
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id<813
SELECT COUNT(*) FROM movie_info,movie_info_idx,title WHERE title.id=movie_info.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id<7 AND movie_info.info_type_id=7
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id=3666
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id<1445767 AND cast_info.role_id<3
SELECT COUNT(*) FROM movie_info_idx WHERE movie_info_idx.info_type_id>100
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.kind_id<7 AND title.production_year=1983
SELECT COUNT(*) FROM movie_info WHERE movie_info.info_type_id>2
SELECT COUNT(*) FROM cast_info,movie_info,title WHERE title.id=cast_info.movie_id AND title.id=movie_info.movie_id AND cast_info.role_id=2 AND movie_info.info_type_id>15
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.production_year<2011
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id=3374181 AND cast_info.role_id>2
SELECT COUNT(*) FROM movie_info WHERE movie_info.info_type_id<8
SELECT COUNT(*) FROM movie_info,movie_info_idx,title WHERE title.id=movie_info.movie_id AND title.id=movie_info_idx.movie_id AND movie_info.info_type_id<3
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id AND movie_keyword.keyword_id<39603
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND movie_keyword.keyword_id>486
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND title.kind_id=1 AND title.production_year>1905 AND movie_info_idx.info_type_id=101
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id=6061
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id AND movie_keyword.keyword_id>5283
SELECT COUNT(*) FROM movie_info WHERE movie_info.info_type_id>13
SELECT COUNT(*) FROM cast_info,movie_companies,title WHERE title.id=movie_companies.movie_id AND title.id=cast_info.movie_id AND title.kind_id=1 AND title.production_year>1985 AND movie_companies.company_id>78 AND movie_companies.company_type_id<2 AND cast_info.person_id>502692
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND title.production_year<2006
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id=6226
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND movie_keyword.keyword_id<3302
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id=1 AND title.production_year<2008 AND movie_companies.company_type_id<2 AND movie_keyword.keyword_id>49177
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND movie_companies.company_id<46
SELECT COUNT(*) FROM movie_companies,movie_info,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info.movie_id
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id>335
SELECT COUNT(*) FROM movie_info WHERE movie_info.info_type_id>6
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.kind_id=7 AND title.production_year<2002
SELECT COUNT(*) FROM movie_info_idx,movie_keyword,title WHERE title.id=movie_info_idx.movie_id AND title.id=movie_keyword.movie_id AND movie_info_idx.info_type_id<101 AND movie_keyword.keyword_id<7168
SELECT COUNT(*) FROM movie_info_idx,movie_keyword,title WHERE title.id=movie_info_idx.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id<7 AND title.production_year>2006
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id>1556
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND title.kind_id<7 AND movie_companies.company_id=109 AND movie_companies.company_type_id<2
SELECT COUNT(*) FROM movie_companies,movie_info,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info.movie_id AND title.kind_id<7 AND title.production_year<1990 AND movie_companies.company_id<708 AND movie_companies.company_type_id=1
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id=196
SELECT COUNT(*) FROM movie_info,movie_keyword,title WHERE title.id=movie_info.movie_id AND title.id=movie_keyword.movie_id AND movie_keyword.keyword_id>3678
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.kind_id>1 AND cast_info.person_id=679131 AND cast_info.role_id<2
SELECT COUNT(*) FROM cast_info WHERE cast_info.role_id<5
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND title.kind_id=1 AND movie_info_idx.info_type_id=100
SELECT COUNT(*) FROM movie_info,movie_info_idx,title WHERE title.id=movie_info.movie_id AND title.id=movie_info_idx.movie_id AND title.production_year>1988 AND movie_info_idx.info_type_id>101
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND title.kind_id>4 AND title.production_year=2009
SELECT COUNT(*) FROM cast_info,movie_info,title WHERE title.id=cast_info.movie_id AND title.id=movie_info.movie_id AND title.kind_id<7 AND cast_info.person_id=988900 AND movie_info.info_type_id=16
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.production_year<2012 AND movie_keyword.keyword_id>47402
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.production_year=1995
SELECT COUNT(*) FROM movie_info,movie_keyword,title WHERE title.id=movie_info.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id<7 AND movie_info.info_type_id>7
SELECT COUNT(*) FROM movie_info,movie_keyword,title WHERE title.id=movie_info.movie_id AND title.id=movie_keyword.movie_id AND title.production_year=1969
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id=386 AND movie_companies.company_type_id<2
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND title.kind_id<7 AND movie_info_idx.info_type_id=99
SELECT COUNT(*) FROM movie_info,movie_info_idx,title WHERE title.id=movie_info.movie_id AND title.id=movie_info_idx.movie_id AND title.production_year>1984 AND movie_info.info_type_id=17
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND title.production_year<1903 AND movie_info_idx.info_type_id>100
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id>3365
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND movie_companies.company_type_id=2
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND title.kind_id<4
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND movie_keyword.keyword_id>1727
SELECT COUNT(*) FROM movie_info_idx,movie_keyword,title WHERE title.id=movie_info_idx.movie_id AND title.id=movie_keyword.movie_id
SELECT COUNT(*) FROM title WHERE title.kind_id=2 AND title.production_year>2013
SELECT COUNT(*) FROM movie_info_idx,movie_keyword,title WHERE title.id=movie_info_idx.movie_id AND title.id=movie_keyword.movie_id AND movie_keyword.keyword_id<53992
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id=80227
SELECT COUNT(*) FROM movie_info,movie_keyword,title WHERE title.id=movie_info.movie_id AND title.id=movie_keyword.movie_id AND movie_info.info_type_id=5 AND movie_keyword.keyword_id=1118
SELECT COUNT(*) FROM title WHERE title.kind_id=1 AND title.production_year>2012
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id=3706835 AND cast_info.role_id>1
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id<22189 AND movie_companies.company_type_id>1
SELECT COUNT(*) FROM movie_companies,movie_info,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info.movie_id AND movie_companies.company_type_id<2
SELECT COUNT(*) FROM movie_info,movie_info_idx,title WHERE title.id=movie_info.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id<7 AND title.production_year>2004 AND movie_info.info_type_id<17
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND title.kind_id<7 AND title.production_year>1980
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id<19732
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id=7777 AND movie_companies.company_type_id=1
SELECT COUNT(*) FROM movie_companies,movie_info_idx,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id<7 AND movie_companies.company_id>4803
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id=61147
SELECT COUNT(*) FROM cast_info,movie_info,title WHERE title.id=cast_info.movie_id AND title.id=movie_info.movie_id AND title.kind_id>3 AND title.production_year=1977
SELECT COUNT(*) FROM cast_info,movie_companies,title WHERE title.id=movie_companies.movie_id AND title.id=cast_info.movie_id AND title.kind_id<2 AND title.production_year<1995 AND cast_info.person_id<237757 AND cast_info.role_id<2
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id<382
SELECT COUNT(*) FROM movie_info,movie_info_idx,title WHERE title.id=movie_info.movie_id AND title.id=movie_info_idx.movie_id AND movie_info_idx.info_type_id>100
SELECT COUNT(*) FROM cast_info,movie_info_idx,title WHERE title.id=cast_info.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id=7
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND title.kind_id=7 AND title.production_year>2007
SELECT COUNT(*) FROM movie_companies,movie_info_idx,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info_idx.movie_id AND movie_companies.company_id<176656 AND movie_info_idx.info_type_id>99
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id<1323260 AND cast_info.role_id=2
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND movie_keyword.keyword_id<3596
SELECT COUNT(*) FROM movie_info,movie_info_idx,title WHERE title.id=movie_info.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id<7
SELECT COUNT(*) FROM movie_info,movie_info_idx,title WHERE title.id=movie_info.movie_id AND title.id=movie_info_idx.movie_id AND title.production_year<1970
SELECT COUNT(*) FROM movie_info,movie_keyword,title WHERE title.id=movie_info.movie_id AND title.id=movie_keyword.movie_id AND title.production_year>1976 AND movie_keyword.keyword_id<32207
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND movie_keyword.keyword_id<2488
SELECT COUNT(*) FROM movie_info_idx,movie_keyword,title WHERE title.id=movie_info_idx.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id=7
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id<2453800 AND cast_info.role_id<5
SELECT COUNT(*) FROM movie_info_idx,movie_keyword,title WHERE title.id=movie_info_idx.movie_id AND title.id=movie_keyword.movie_id AND title.production_year>1966 AND movie_info_idx.info_type_id<100
SELECT COUNT(*) FROM movie_info,movie_keyword,title WHERE title.id=movie_info.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id<7 AND movie_info.info_type_id=16
SELECT COUNT(*) FROM movie_info WHERE movie_info.info_type_id=6
SELECT COUNT(*) FROM cast_info,movie_info_idx,title WHERE title.id=cast_info.movie_id AND title.id=movie_info_idx.movie_id AND title.production_year<1915 AND movie_info_idx.info_type_id=101
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND movie_keyword.keyword_id>31525
SELECT COUNT(*) FROM movie_companies,movie_info_idx,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id<2 AND movie_companies.company_type_id=1
SELECT COUNT(*) FROM cast_info,movie_keyword,title WHERE title.id=cast_info.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id=7 AND title.production_year<2009 AND cast_info.role_id<3
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.kind_id=7 AND title.production_year>1950
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.kind_id=7 AND title.production_year=2008
SELECT COUNT(*) FROM movie_companies,movie_info_idx,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id<7 AND title.production_year<2004
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND title.kind_id>1 AND title.production_year<2002 AND movie_companies.company_id=192 AND movie_companies.company_type_id<2
SELECT COUNT(*) FROM movie_info WHERE movie_info.info_type_id=65
SELECT COUNT(*) FROM cast_info WHERE cast_info.role_id>10
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.kind_id>1 AND title.production_year=1961 AND cast_info.role_id>8
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.kind_id=7 AND movie_info.info_type_id<8
SELECT COUNT(*) FROM title WHERE title.kind_id>1 AND title.production_year>2007
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND title.kind_id<7 AND title.production_year>1989
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.production_year=1988
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND title.kind_id<7 AND title.production_year<2013 AND movie_info_idx.info_type_id=99
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND title.kind_id=1 AND movie_companies.company_id>11930
SELECT COUNT(*) FROM title WHERE title.kind_id=7 AND title.production_year=2013
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND title.kind_id=7 AND title.production_year>1990 AND movie_info_idx.info_type_id=100
SELECT COUNT(*) FROM cast_info,movie_keyword,title WHERE title.id=cast_info.movie_id AND title.id=movie_keyword.movie_id AND movie_keyword.keyword_id<13738
SELECT COUNT(*) FROM movie_info,movie_keyword,title WHERE title.id=movie_info.movie_id AND title.id=movie_keyword.movie_id AND movie_info.info_type_id>1 AND movie_keyword.keyword_id<27197
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id<12321
SELECT COUNT(*) FROM movie_info WHERE movie_info.info_type_id>8
SELECT COUNT(*) FROM movie_info,movie_keyword,title WHERE title.id=movie_info.movie_id AND title.id=movie_keyword.movie_id AND movie_keyword.keyword_id=72112
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND movie_info_idx.info_type_id<101
SELECT COUNT(*) FROM cast_info,movie_companies,title WHERE title.id=movie_companies.movie_id AND title.id=cast_info.movie_id AND title.production_year>2005
SELECT COUNT(*) FROM title WHERE title.production_year<2008
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id>4436
SELECT COUNT(*) FROM cast_info,movie_info,title WHERE title.id=cast_info.movie_id AND title.id=movie_info.movie_id AND title.production_year<1983 AND cast_info.role_id=3 AND movie_info.info_type_id>17
SELECT COUNT(*) FROM movie_info WHERE movie_info.info_type_id=66
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id=16264
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.production_year>1959
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND title.kind_id<7 AND title.production_year=2012 AND movie_companies.company_id<47789
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.kind_id<6 AND title.production_year=1952
SELECT COUNT(*) FROM movie_info,movie_info_idx,title WHERE title.id=movie_info.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id<7 AND title.production_year<1984 AND movie_info.info_type_id<16
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id AND movie_companies.company_id>4599 AND movie_keyword.keyword_id=1531
SELECT COUNT(*) FROM cast_info,movie_info_idx,title WHERE title.id=cast_info.movie_id AND title.id=movie_info_idx.movie_id AND cast_info.person_id>2160118 AND cast_info.role_id>2 AND movie_info_idx.info_type_id=99
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND movie_info.info_type_id>2
SELECT COUNT(*) FROM movie_info WHERE movie_info.info_type_id>4
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id AND movie_companies.company_type_id=1
SELECT COUNT(*) FROM movie_info WHERE movie_info.info_type_id<106
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND title.kind_id<2 AND title.production_year>2010
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.kind_id<7
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id>270
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND title.kind_id<7 AND title.production_year<1912 AND movie_info_idx.info_type_id<100
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND title.production_year=2000 AND movie_info_idx.info_type_id>100
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id AND movie_companies.company_id<27627
SELECT COUNT(*) FROM movie_info WHERE movie_info.info_type_id>3
SELECT COUNT(*) FROM cast_info,movie_keyword,title WHERE title.id=cast_info.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id=7 AND title.production_year<1951 AND cast_info.person_id<891747
SELECT COUNT(*) FROM cast_info,movie_info_idx,title WHERE title.id=cast_info.movie_id AND title.id=movie_info_idx.movie_id AND title.production_year<2012 AND movie_info_idx.info_type_id>101
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND title.production_year<2005 AND movie_info_idx.info_type_id=99
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND title.kind_id=2 AND movie_info_idx.info_type_id=100
