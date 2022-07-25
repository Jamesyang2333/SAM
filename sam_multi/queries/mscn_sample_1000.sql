SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id<738
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND cast_info.person_id=2395101 AND cast_info.role_id=2
SELECT COUNT(*) FROM movie_info_idx,movie_keyword,title WHERE title.id=movie_info_idx.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id>1 AND title.production_year=2012
SELECT COUNT(*) FROM cast_info,movie_info_idx,title WHERE title.id=cast_info.movie_id AND title.id=movie_info_idx.movie_id AND title.production_year<1913 AND cast_info.role_id=1
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND movie_companies.company_id=4359
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id AND title.production_year=2008 AND movie_companies.company_type_id>1 AND movie_keyword.keyword_id>90364
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND title.production_year=2006 AND movie_companies.company_type_id=2
SELECT COUNT(*) FROM title WHERE title.kind_id=7 AND title.production_year=1958
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND title.kind_id=1 AND title.production_year<2000 AND movie_info_idx.info_type_id<101
SELECT COUNT(*) FROM cast_info,movie_companies,title WHERE title.id=movie_companies.movie_id AND title.id=cast_info.movie_id AND title.kind_id=7 AND title.production_year>1911 AND movie_companies.company_type_id<2
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.kind_id<7 AND movie_keyword.keyword_id>4601
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id>3592850
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id<2745737 AND cast_info.role_id>10
SELECT COUNT(*) FROM title WHERE title.production_year>1954
SELECT COUNT(*) FROM cast_info,movie_info,title WHERE title.id=cast_info.movie_id AND title.id=movie_info.movie_id AND title.kind_id=7 AND title.production_year<2003 AND cast_info.person_id>2026559
SELECT COUNT(*) FROM cast_info,movie_info_idx,title WHERE title.id=cast_info.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id<7 AND title.production_year>1964
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND movie_companies.company_id>11203
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND title.kind_id<3 AND title.production_year=2007
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id=80801 AND cast_info.role_id=3
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id<86101 AND movie_companies.company_type_id=1
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id>2039555
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id>4032065
SELECT COUNT(*) FROM cast_info,movie_companies,title WHERE title.id=movie_companies.movie_id AND title.id=cast_info.movie_id AND movie_companies.company_id=6 AND movie_companies.company_type_id<2
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND cast_info.person_id>1013210 AND cast_info.role_id<3
SELECT COUNT(*) FROM cast_info,movie_companies,title WHERE title.id=movie_companies.movie_id AND title.id=cast_info.movie_id AND movie_companies.company_id<23930 AND movie_companies.company_type_id=1 AND cast_info.person_id<164507
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND movie_companies.company_id<467 AND movie_companies.company_type_id>1
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id<2986
SELECT COUNT(*) FROM cast_info,movie_companies,title WHERE title.id=movie_companies.movie_id AND title.id=cast_info.movie_id AND movie_companies.company_id>88711
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND movie_keyword.keyword_id<38808
SELECT COUNT(*) FROM movie_companies,movie_info_idx,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id=7 AND title.production_year>2009 AND movie_companies.company_id>35769
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.kind_id=1 AND title.production_year<2005 AND movie_info.info_type_id>4
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND movie_companies.company_id<44257 AND movie_companies.company_type_id<2
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.production_year=2011 AND movie_keyword.keyword_id=34308
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.production_year=1952
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND title.kind_id=7 AND title.production_year=2010 AND movie_info_idx.info_type_id>99
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id<2014561 AND cast_info.role_id=1
SELECT COUNT(*) FROM movie_companies,movie_info,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info.movie_id AND movie_companies.company_id<15706
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.production_year>2001 AND movie_keyword.keyword_id=3086
SELECT COUNT(*) FROM movie_info,movie_info_idx,title WHERE title.id=movie_info.movie_id AND title.id=movie_info_idx.movie_id AND title.production_year<1960
SELECT COUNT(*) FROM movie_info,movie_keyword,title WHERE title.id=movie_info.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id<2 AND title.production_year=2012 AND movie_keyword.keyword_id=11540
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.kind_id>1 AND title.production_year<1990
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id=2974
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.kind_id=2 AND title.production_year<1983
SELECT COUNT(*) FROM movie_companies,movie_info,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info.movie_id AND movie_companies.company_id>1129 AND movie_companies.company_type_id=2 AND movie_info.info_type_id=8
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.production_year>0
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id=7654
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id<17363
SELECT COUNT(*) FROM movie_info,movie_info_idx,title WHERE title.id=movie_info.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id<7 AND title.production_year>2011 AND movie_info.info_type_id=3 AND movie_info_idx.info_type_id>100
SELECT COUNT(*) FROM title WHERE title.kind_id<2 AND title.production_year=1994
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id>3276337 AND cast_info.role_id=10
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id>632 AND movie_companies.company_type_id=1
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND title.kind_id<4 AND title.production_year<2005
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id=3761
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id=1401
SELECT COUNT(*) FROM movie_info_idx,movie_keyword,title WHERE title.id=movie_info_idx.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id=7 AND title.production_year>1976 AND movie_info_idx.info_type_id=99
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND title.kind_id<4 AND title.production_year<1980 AND movie_companies.company_id=14490 AND movie_companies.company_type_id=2
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id>742 AND movie_companies.company_type_id=2
SELECT COUNT(*) FROM movie_companies,movie_info_idx,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info_idx.movie_id AND title.production_year>1964 AND movie_companies.company_type_id<2
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND cast_info.person_id>1943841 AND cast_info.role_id>4
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.production_year>1979
SELECT COUNT(*) FROM title WHERE title.kind_id=3 AND title.production_year<2010
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND movie_keyword.keyword_id<10468
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.kind_id=4 AND title.production_year<1979
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id>8449
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.kind_id>3 AND title.production_year=2006 AND cast_info.person_id<730793 AND cast_info.role_id=1
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND movie_companies.company_id=11208 AND movie_companies.company_type_id>1
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id=80227
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.production_year>2010 AND movie_keyword.keyword_id=1741
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id>93
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id>464
SELECT COUNT(*) FROM movie_info,movie_keyword,title WHERE title.id=movie_info.movie_id AND title.id=movie_keyword.movie_id AND movie_info.info_type_id>3 AND movie_keyword.keyword_id>49314
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id>118585 AND cast_info.role_id=2
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id<5827
SELECT COUNT(*) FROM movie_info_idx,movie_keyword,title WHERE title.id=movie_info_idx.movie_id AND title.id=movie_keyword.movie_id AND movie_info_idx.info_type_id<100 AND movie_keyword.keyword_id<2931
SELECT COUNT(*) FROM movie_info_idx,movie_keyword,title WHERE title.id=movie_info_idx.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id=7 AND title.production_year<2006 AND movie_info_idx.info_type_id=101
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.production_year=2000 AND movie_info.info_type_id<16
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id<15944
SELECT COUNT(*) FROM cast_info,movie_keyword,title WHERE title.id=cast_info.movie_id AND title.id=movie_keyword.movie_id AND cast_info.person_id<3992104 AND movie_keyword.keyword_id>196
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id>671041 AND cast_info.role_id>4
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.kind_id<7 AND title.production_year=2006 AND cast_info.person_id>1303580 AND cast_info.role_id=1
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND movie_companies.company_id>12549 AND movie_companies.company_type_id=1
SELECT COUNT(*) FROM cast_info,movie_info,title WHERE title.id=cast_info.movie_id AND title.id=movie_info.movie_id AND title.production_year<2012 AND cast_info.person_id<303375 AND movie_info.info_type_id=8
SELECT COUNT(*) FROM title WHERE title.kind_id>2 AND title.production_year<1925
SELECT COUNT(*) FROM cast_info,movie_info,title WHERE title.id=cast_info.movie_id AND title.id=movie_info.movie_id AND cast_info.person_id<1217527
SELECT COUNT(*) FROM movie_companies,movie_info_idx,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info_idx.movie_id AND movie_companies.company_id>16508 AND movie_companies.company_type_id>1
SELECT COUNT(*) FROM movie_info,movie_keyword,title WHERE title.id=movie_info.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id=7 AND title.production_year>1991 AND movie_info.info_type_id>9
SELECT COUNT(*) FROM cast_info,movie_info,title WHERE title.id=cast_info.movie_id AND title.id=movie_info.movie_id AND cast_info.role_id=10 AND movie_info.info_type_id>6
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.production_year<2009 AND cast_info.person_id>1065391
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.kind_id=7 AND title.production_year=2012 AND movie_info.info_type_id>15
SELECT COUNT(*) FROM movie_companies,movie_info_idx,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id<7 AND movie_companies.company_id>20 AND movie_info_idx.info_type_id>100
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id<13976 AND movie_companies.company_type_id<2
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.kind_id=3 AND cast_info.person_id>21321 AND cast_info.role_id=8
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND title.production_year>1993 AND movie_companies.company_id=8812 AND movie_companies.company_type_id=2
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id>769 AND movie_companies.company_type_id=1
SELECT COUNT(*) FROM title WHERE title.kind_id=2 AND title.production_year=1995
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id AND title.production_year<1988 AND movie_keyword.keyword_id>3806
SELECT COUNT(*) FROM cast_info,movie_info,title WHERE title.id=cast_info.movie_id AND title.id=movie_info.movie_id AND title.kind_id=7 AND cast_info.person_id>2334699 AND cast_info.role_id<10 AND movie_info.info_type_id<4
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id=35734
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id=2628586
SELECT COUNT(*) FROM cast_info,movie_keyword,title WHERE title.id=cast_info.movie_id AND title.id=movie_keyword.movie_id AND cast_info.person_id>21068 AND cast_info.role_id=1
SELECT COUNT(*) FROM cast_info,movie_companies,title WHERE title.id=movie_companies.movie_id AND title.id=cast_info.movie_id AND title.kind_id>1 AND cast_info.role_id<8
SELECT COUNT(*) FROM movie_info,movie_keyword,title WHERE title.id=movie_info.movie_id AND title.id=movie_keyword.movie_id AND title.production_year=1965 AND movie_info.info_type_id=16
SELECT COUNT(*) FROM cast_info,movie_info,title WHERE title.id=cast_info.movie_id AND title.id=movie_info.movie_id AND title.kind_id=7 AND title.production_year=2012
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.kind_id=7 AND title.production_year>2003 AND movie_info.info_type_id<16
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id<1460791 AND cast_info.role_id>10
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id<1113690 AND cast_info.role_id=1
SELECT COUNT(*) FROM cast_info,movie_info,title WHERE title.id=cast_info.movie_id AND title.id=movie_info.movie_id AND cast_info.person_id=476494 AND cast_info.role_id>1
SELECT COUNT(*) FROM movie_companies,movie_info_idx,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info_idx.movie_id AND movie_companies.company_id>72131
SELECT COUNT(*) FROM movie_companies,movie_info_idx,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info_idx.movie_id AND title.production_year<2012 AND movie_companies.company_id>30665 AND movie_companies.company_type_id>1
SELECT COUNT(*) FROM cast_info,movie_companies,title WHERE title.id=movie_companies.movie_id AND title.id=cast_info.movie_id AND cast_info.person_id>1986690 AND cast_info.role_id>2
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id<43991
SELECT COUNT(*) FROM movie_companies,movie_info,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info.movie_id AND movie_companies.company_type_id>1 AND movie_info.info_type_id=4
SELECT COUNT(*) FROM movie_info,movie_info_idx,title WHERE title.id=movie_info.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id<7 AND title.production_year<1996 AND movie_info_idx.info_type_id>100
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.kind_id=3 AND title.production_year=1997 AND cast_info.person_id<1039919
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.kind_id=7 AND title.production_year=2006 AND movie_info.info_type_id=8
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id>2631
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id AND movie_companies.company_id=8887 AND movie_keyword.keyword_id>2488
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id AND movie_keyword.keyword_id=117
SELECT COUNT(*) FROM cast_info,movie_keyword,title WHERE title.id=cast_info.movie_id AND title.id=movie_keyword.movie_id AND movie_keyword.keyword_id>6568
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id<1278590 AND cast_info.role_id>3
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id=460
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id<15771
SELECT COUNT(*) FROM movie_info,movie_keyword,title WHERE title.id=movie_info.movie_id AND title.id=movie_keyword.movie_id AND movie_info.info_type_id=16 AND movie_keyword.keyword_id<2554
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id<1310
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id<16190
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND title.production_year>2005 AND movie_companies.company_id<49
SELECT COUNT(*) FROM cast_info,movie_info_idx,title WHERE title.id=cast_info.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id<7 AND cast_info.person_id>1347087 AND cast_info.role_id<4 AND movie_info_idx.info_type_id=100
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.kind_id=7 AND title.production_year=2012 AND movie_info.info_type_id>7
SELECT COUNT(*) FROM movie_info_idx,movie_keyword,title WHERE title.id=movie_info_idx.movie_id AND title.id=movie_keyword.movie_id AND title.production_year>2009 AND movie_info_idx.info_type_id>101 AND movie_keyword.keyword_id>10080
SELECT COUNT(*) FROM movie_companies,movie_info_idx,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id<7 AND title.production_year<2006 AND movie_companies.company_type_id<2
SELECT COUNT(*) FROM movie_info_idx,movie_keyword,title WHERE title.id=movie_info_idx.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id<7 AND title.production_year<2010 AND movie_info_idx.info_type_id<100
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id>47090
SELECT COUNT(*) FROM cast_info,movie_companies,title WHERE title.id=movie_companies.movie_id AND title.id=cast_info.movie_id AND title.kind_id=7 AND title.production_year>2006 AND movie_companies.company_id=2416
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND cast_info.person_id=2895323
SELECT COUNT(*) FROM cast_info,movie_info,title WHERE title.id=cast_info.movie_id AND title.id=movie_info.movie_id AND title.production_year>1978 AND cast_info.role_id>1
SELECT COUNT(*) FROM movie_info_idx,movie_keyword,title WHERE title.id=movie_info_idx.movie_id AND title.id=movie_keyword.movie_id AND title.production_year>2010 AND movie_info_idx.info_type_id<100
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.kind_id>2 AND title.production_year=2010
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.kind_id<7 AND title.production_year<1955
SELECT COUNT(*) FROM cast_info,movie_companies,title WHERE title.id=movie_companies.movie_id AND title.id=cast_info.movie_id AND title.kind_id<7 AND title.production_year=2007 AND movie_companies.company_type_id<2 AND cast_info.person_id<3623245 AND cast_info.role_id<8
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND title.kind_id>1 AND title.production_year>1945 AND movie_info_idx.info_type_id=101
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id>7663
SELECT COUNT(*) FROM movie_info,movie_info_idx,title WHERE title.id=movie_info.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id=7 AND title.production_year=2000 AND movie_info.info_type_id>16
SELECT COUNT(*) FROM cast_info,movie_info_idx,title WHERE title.id=cast_info.movie_id AND title.id=movie_info_idx.movie_id AND cast_info.role_id<2 AND movie_info_idx.info_type_id=100
SELECT COUNT(*) FROM movie_info_idx,movie_keyword,title WHERE title.id=movie_info_idx.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id<7 AND title.production_year>2010 AND movie_keyword.keyword_id>28791
SELECT COUNT(*) FROM title WHERE title.production_year=2006
SELECT COUNT(*) FROM movie_info,movie_keyword,title WHERE title.id=movie_info.movie_id AND title.id=movie_keyword.movie_id AND title.production_year>2009 AND movie_info.info_type_id>3 AND movie_keyword.keyword_id>3658
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND cast_info.person_id<2979987 AND cast_info.role_id=4
SELECT COUNT(*) FROM cast_info,movie_info,title WHERE title.id=cast_info.movie_id AND title.id=movie_info.movie_id AND title.kind_id<3 AND title.production_year<1964 AND cast_info.role_id=1
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.production_year=2009 AND movie_info.info_type_id=8
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND title.production_year=2011 AND movie_companies.company_id=990
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.kind_id=7 AND title.production_year<2011 AND cast_info.person_id=3589533 AND cast_info.role_id>4
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND title.kind_id>3 AND title.production_year<2011 AND movie_companies.company_type_id>1
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.kind_id>1 AND title.production_year>2004
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.production_year=2011 AND movie_info.info_type_id<4
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND movie_companies.company_id=1877 AND movie_companies.company_type_id>1
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id<28221
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id>339
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND movie_keyword.keyword_id=30246
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND movie_companies.company_id=139865 AND movie_companies.company_type_id=2
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND movie_companies.company_id=77219
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id AND movie_companies.company_id>134642 AND movie_companies.company_type_id>1 AND movie_keyword.keyword_id<37234
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id=16803
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND title.kind_id>2 AND title.production_year<1946 AND movie_info_idx.info_type_id>100
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.production_year<2000 AND cast_info.person_id>214193 AND cast_info.role_id=4
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND title.kind_id=1 AND title.production_year>2001 AND movie_info_idx.info_type_id<100
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND title.kind_id=1 AND title.production_year>2006
SELECT COUNT(*) FROM movie_companies,movie_info,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info.movie_id AND title.kind_id=1 AND title.production_year=1972 AND movie_companies.company_type_id=1
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id<3 AND title.production_year>2001
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.production_year<1963
SELECT COUNT(*) FROM cast_info,movie_keyword,title WHERE title.id=cast_info.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id<7 AND title.production_year<1995
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id=4853
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id>16547 AND movie_companies.company_type_id>1
SELECT COUNT(*) FROM movie_info,movie_keyword,title WHERE title.id=movie_info.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id>4 AND title.production_year<1992 AND movie_info.info_type_id<8
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND movie_keyword.keyword_id>5299
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id<66
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id=78986 AND movie_companies.company_type_id=2
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id=10336
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.kind_id=4 AND title.production_year>1996
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND title.production_year<2008 AND movie_info_idx.info_type_id>100
SELECT COUNT(*) FROM cast_info,movie_info_idx,title WHERE title.id=cast_info.movie_id AND title.id=movie_info_idx.movie_id AND cast_info.role_id=3
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id>75058 AND movie_companies.company_type_id=2
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND movie_companies.company_id=11588
SELECT COUNT(*) FROM movie_companies,movie_info_idx,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info_idx.movie_id AND title.production_year=2000 AND movie_info_idx.info_type_id<101
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id<31341
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id>1084
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.production_year>1987
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.production_year>2002 AND movie_keyword.keyword_id=3673
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND title.kind_id=7 AND title.production_year>2006
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.kind_id<7 AND title.production_year<1997
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND title.production_year=2006 AND movie_info_idx.info_type_id=101
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND title.kind_id<7 AND title.production_year>2001 AND movie_info_idx.info_type_id>99
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id<2053142 AND cast_info.role_id>4
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id>4 AND title.production_year>1923 AND movie_companies.company_id>87348 AND movie_companies.company_type_id>1
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.kind_id=1 AND title.production_year>2010 AND movie_keyword.keyword_id>2128
SELECT COUNT(*) FROM movie_info,movie_info_idx,title WHERE title.id=movie_info.movie_id AND title.id=movie_info_idx.movie_id AND title.production_year=1975 AND movie_info.info_type_id=16
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.kind_id<2 AND movie_info.info_type_id>4
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.kind_id<7 AND title.production_year=1991
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.production_year>1998 AND movie_keyword.keyword_id<196
SELECT COUNT(*) FROM movie_info_idx,movie_keyword,title WHERE title.id=movie_info_idx.movie_id AND title.id=movie_keyword.movie_id AND title.production_year<2007 AND movie_info_idx.info_type_id<101 AND movie_keyword.keyword_id>11577
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.kind_id<7 AND cast_info.person_id=975230
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND movie_companies.company_id<7851
SELECT COUNT(*) FROM movie_companies,movie_info_idx,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id>4 AND movie_companies.company_id<91411 AND movie_companies.company_type_id<2
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.production_year=1997 AND movie_info.info_type_id<2
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id=832
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.kind_id=4 AND title.production_year>1985 AND movie_keyword.keyword_id>16131
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id<480 AND movie_companies.company_type_id<2
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND title.kind_id=4 AND title.production_year=1973
SELECT COUNT(*) FROM cast_info,movie_companies,title WHERE title.id=movie_companies.movie_id AND title.id=cast_info.movie_id AND movie_companies.company_id>28339 AND cast_info.person_id=1123582 AND cast_info.role_id>2
SELECT COUNT(*) FROM cast_info,movie_info,title WHERE title.id=cast_info.movie_id AND title.id=movie_info.movie_id AND title.kind_id=1 AND title.production_year=2009 AND cast_info.person_id>1111309 AND cast_info.role_id=1
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.kind_id<7 AND title.production_year=2007 AND cast_info.role_id<3
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id<106835
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id>2372971 AND cast_info.role_id>2
SELECT COUNT(*) FROM movie_companies,movie_info,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info.movie_id AND movie_companies.company_id=533 AND movie_companies.company_type_id=2 AND movie_info.info_type_id=3
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND cast_info.role_id<4
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id<7639
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id<6115
SELECT COUNT(*) FROM movie_companies,movie_info_idx,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info_idx.movie_id AND movie_companies.company_id>11705 AND movie_companies.company_type_id=1 AND movie_info_idx.info_type_id=100
SELECT COUNT(*) FROM movie_info,movie_keyword,title WHERE title.id=movie_info.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id=7 AND movie_keyword.keyword_id<59118
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id=1036
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND movie_companies.company_id<150
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id>32911
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.production_year=1968 AND movie_keyword.keyword_id>43067
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND title.kind_id=7 AND title.production_year=1962 AND movie_info_idx.info_type_id<101
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id<6646
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id=26858 AND movie_companies.company_type_id=1
SELECT COUNT(*) FROM cast_info,movie_info,title WHERE title.id=cast_info.movie_id AND title.id=movie_info.movie_id AND title.kind_id>1 AND title.production_year=2006 AND cast_info.role_id>2 AND movie_info.info_type_id<98
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.kind_id>1 AND title.production_year>1999 AND movie_info.info_type_id<7
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id>20787 AND movie_companies.company_type_id<2
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND movie_companies.company_id>5
SELECT COUNT(*) FROM movie_companies,movie_info_idx,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info_idx.movie_id AND title.production_year<1996 AND movie_info_idx.info_type_id=100
SELECT COUNT(*) FROM movie_companies,movie_info,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info.movie_id AND title.kind_id=4 AND title.production_year=2004 AND movie_companies.company_id>221 AND movie_companies.company_type_id<2
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.kind_id>1 AND title.production_year>1974
SELECT COUNT(*) FROM movie_info,movie_info_idx,title WHERE title.id=movie_info.movie_id AND title.id=movie_info_idx.movie_id AND title.production_year<2010 AND movie_info.info_type_id<16 AND movie_info_idx.info_type_id<100
SELECT COUNT(*) FROM movie_companies,movie_info,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info.movie_id AND title.kind_id<7 AND movie_companies.company_id>689 AND movie_companies.company_type_id<2
SELECT COUNT(*) FROM cast_info,movie_info,title WHERE title.id=cast_info.movie_id AND title.id=movie_info.movie_id AND title.kind_id>1 AND cast_info.person_id<1972511 AND cast_info.role_id>10
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id=1971480
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.kind_id=1 AND title.production_year>2007
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id<4448
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id=33638 AND movie_companies.company_type_id>1
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND movie_companies.company_id<44090
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id=2578
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND movie_companies.company_id>11387
SELECT COUNT(*) FROM movie_companies,movie_info_idx,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info_idx.movie_id AND title.production_year>1991 AND movie_companies.company_id>2561
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id<1625189 AND cast_info.role_id=2
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id=11916 AND movie_companies.company_type_id=1
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id<45383 AND movie_companies.company_type_id<2
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND cast_info.person_id>2344664 AND cast_info.role_id>10
SELECT COUNT(*) FROM movie_info_idx,movie_keyword,title WHERE title.id=movie_info_idx.movie_id AND title.id=movie_keyword.movie_id AND title.production_year>1991 AND movie_info_idx.info_type_id=99
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.production_year>1913 AND movie_keyword.keyword_id=6431
SELECT COUNT(*) FROM cast_info,movie_info_idx,title WHERE title.id=cast_info.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id=1 AND title.production_year<2006 AND cast_info.role_id<3 AND movie_info_idx.info_type_id>101
SELECT COUNT(*) FROM movie_info,movie_keyword,title WHERE title.id=movie_info.movie_id AND title.id=movie_keyword.movie_id AND title.production_year=2012 AND movie_keyword.keyword_id>1787
SELECT COUNT(*) FROM movie_companies,movie_info,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info.movie_id AND title.production_year=2013 AND movie_companies.company_id>10584 AND movie_companies.company_type_id=2
SELECT COUNT(*) FROM cast_info,movie_info,title WHERE title.id=cast_info.movie_id AND title.id=movie_info.movie_id AND title.production_year<1985 AND cast_info.role_id>3
SELECT COUNT(*) FROM cast_info,movie_companies,title WHERE title.id=movie_companies.movie_id AND title.id=cast_info.movie_id AND movie_companies.company_type_id>1 AND cast_info.person_id<401304
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id<17704
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND title.production_year=2012
SELECT COUNT(*) FROM cast_info,movie_info,title WHERE title.id=cast_info.movie_id AND title.id=movie_info.movie_id AND title.production_year=2010 AND movie_info.info_type_id<13
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.kind_id<4 AND title.production_year>1991
SELECT COUNT(*) FROM movie_companies,movie_info,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info.movie_id AND movie_companies.company_id<4226 AND movie_info.info_type_id<8
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id>2892
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.production_year<1958 AND cast_info.role_id<10
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id=12218
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id<23532 AND movie_companies.company_type_id=2
SELECT COUNT(*) FROM movie_companies,movie_info_idx,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id<7 AND title.production_year>2003 AND movie_companies.company_id>142799 AND movie_companies.company_type_id=2 AND movie_info_idx.info_type_id>101
SELECT COUNT(*) FROM cast_info,movie_keyword,title WHERE title.id=cast_info.movie_id AND title.id=movie_keyword.movie_id AND cast_info.person_id>2693073 AND movie_keyword.keyword_id=117
SELECT COUNT(*) FROM cast_info,movie_info_idx,title WHERE title.id=cast_info.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id=7 AND title.production_year=2004 AND cast_info.person_id<982750 AND cast_info.role_id=1 AND movie_info_idx.info_type_id>99
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.kind_id=4 AND movie_keyword.keyword_id=3067
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id AND movie_companies.company_id>143949 AND movie_keyword.keyword_id<21346
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND movie_keyword.keyword_id=432
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.production_year=2003 AND movie_keyword.keyword_id<64187
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id<2816973
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.kind_id<7 AND title.production_year=1998
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.kind_id=1 AND title.production_year<1979
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND title.kind_id<7 AND title.production_year<1961 AND movie_info_idx.info_type_id>100
SELECT COUNT(*) FROM cast_info,movie_info,title WHERE title.id=cast_info.movie_id AND title.id=movie_info.movie_id AND title.kind_id<7 AND title.production_year=2011 AND cast_info.role_id>9
SELECT COUNT(*) FROM movie_companies,movie_info_idx,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id<7 AND movie_companies.company_id<36079 AND movie_info_idx.info_type_id>99
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.kind_id=7 AND title.production_year<2007 AND movie_info.info_type_id>2
SELECT COUNT(*) FROM cast_info,movie_info,title WHERE title.id=cast_info.movie_id AND title.id=movie_info.movie_id AND cast_info.person_id=2174927 AND cast_info.role_id>1
SELECT COUNT(*) FROM movie_info,movie_keyword,title WHERE title.id=movie_info.movie_id AND title.id=movie_keyword.movie_id AND movie_info.info_type_id>3 AND movie_keyword.keyword_id=995
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id<15491 AND movie_companies.company_type_id>1
SELECT COUNT(*) FROM cast_info,movie_info,title WHERE title.id=cast_info.movie_id AND title.id=movie_info.movie_id AND title.kind_id<7 AND title.production_year>1942 AND cast_info.person_id<1327307
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id>77866 AND movie_companies.company_type_id>1
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id<189 AND movie_companies.company_type_id<2
SELECT COUNT(*) FROM cast_info,movie_companies,title WHERE title.id=movie_companies.movie_id AND title.id=cast_info.movie_id AND title.production_year=2004 AND cast_info.role_id=2
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id AND title.production_year>1969
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.kind_id=1 AND movie_keyword.keyword_id=353
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id>3014443 AND cast_info.role_id<7
SELECT COUNT(*) FROM cast_info,movie_info_idx,title WHERE title.id=cast_info.movie_id AND title.id=movie_info_idx.movie_id AND movie_info_idx.info_type_id<101
SELECT COUNT(*) FROM movie_info_idx,movie_keyword,title WHERE title.id=movie_info_idx.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id<7 AND title.production_year<1976 AND movie_keyword.keyword_id>2488
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.production_year=1973 AND movie_info.info_type_id>3
SELECT COUNT(*) FROM cast_info,movie_keyword,title WHERE title.id=cast_info.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id>3 AND cast_info.person_id<1336805 AND cast_info.role_id<5 AND movie_keyword.keyword_id>42182
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id AND title.production_year>2003 AND movie_keyword.keyword_id>2358
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.production_year>1969
SELECT COUNT(*) FROM cast_info,movie_info_idx,title WHERE title.id=cast_info.movie_id AND title.id=movie_info_idx.movie_id AND title.production_year=1980 AND cast_info.person_id<4025231 AND cast_info.role_id>1
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND title.kind_id=1 AND title.production_year>1928 AND movie_companies.company_id=70607
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND title.kind_id=7 AND title.production_year>1983 AND movie_info_idx.info_type_id>100
SELECT COUNT(*) FROM movie_info,movie_info_idx,title WHERE title.id=movie_info.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id>1 AND title.production_year<1990 AND movie_info.info_type_id<4 AND movie_info_idx.info_type_id<101
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.kind_id=7 AND cast_info.person_id<3842564 AND cast_info.role_id=8
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND cast_info.person_id=2378244 AND cast_info.role_id>1
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id<1752
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND movie_keyword.keyword_id<20536
SELECT COUNT(*) FROM cast_info WHERE cast_info.role_id<9
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND title.kind_id<7 AND title.production_year>1961
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.kind_id>1 AND title.production_year<1992
SELECT COUNT(*) FROM movie_info_idx,movie_keyword,title WHERE title.id=movie_info_idx.movie_id AND title.id=movie_keyword.movie_id AND movie_keyword.keyword_id<797
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND title.production_year<1967 AND movie_info_idx.info_type_id=101
SELECT COUNT(*) FROM cast_info,movie_info_idx,title WHERE title.id=cast_info.movie_id AND title.id=movie_info_idx.movie_id AND cast_info.person_id>139631 AND cast_info.role_id=1 AND movie_info_idx.info_type_id=101
SELECT COUNT(*) FROM title WHERE title.kind_id>3 AND title.production_year=2011
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.kind_id>1 AND title.production_year>2003 AND movie_keyword.keyword_id=4555
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id<145 AND movie_companies.company_type_id>1
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id>1 AND movie_keyword.keyword_id<126878
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.kind_id>2 AND title.production_year>1997
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id=2753069
SELECT COUNT(*) FROM movie_info_idx,movie_keyword,title WHERE title.id=movie_info_idx.movie_id AND title.id=movie_keyword.movie_id AND title.production_year=2007 AND movie_info_idx.info_type_id=100 AND movie_keyword.keyword_id=1714
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id<2692321 AND cast_info.role_id<6
SELECT COUNT(*) FROM cast_info,movie_info_idx,title WHERE title.id=cast_info.movie_id AND title.id=movie_info_idx.movie_id AND cast_info.person_id>3747415 AND cast_info.role_id>1 AND movie_info_idx.info_type_id<101
SELECT COUNT(*) FROM movie_companies,movie_info_idx,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id=7 AND title.production_year<1972
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND title.kind_id<7 AND title.production_year<1977
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id AND title.production_year=2001 AND movie_companies.company_id>2473 AND movie_keyword.keyword_id<323
SELECT COUNT(*) FROM movie_companies,movie_info_idx,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info_idx.movie_id AND movie_companies.company_id<110490 AND movie_companies.company_type_id>1 AND movie_info_idx.info_type_id>101
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id=79228
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id>989134 AND cast_info.role_id=1
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id=1695
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.kind_id<7 AND cast_info.person_id>2371775 AND cast_info.role_id>1
SELECT COUNT(*) FROM movie_info,movie_keyword,title WHERE title.id=movie_info.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id>1 AND title.production_year=2002 AND movie_keyword.keyword_id<114
SELECT COUNT(*) FROM movie_info_idx,movie_keyword,title WHERE title.id=movie_info_idx.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id=7 AND title.production_year<2004 AND movie_keyword.keyword_id<41460
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND movie_keyword.keyword_id<21125
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id=1236 AND movie_companies.company_type_id=2
SELECT COUNT(*) FROM cast_info,movie_info,title WHERE title.id=cast_info.movie_id AND title.id=movie_info.movie_id AND cast_info.person_id>759589 AND cast_info.role_id<4
SELECT COUNT(*) FROM movie_companies,movie_info_idx,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id>1 AND title.production_year=2007 AND movie_companies.company_type_id>1 AND movie_info_idx.info_type_id=100
SELECT COUNT(*) FROM movie_info,movie_info_idx,title WHERE title.id=movie_info.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id=3 AND title.production_year=2011
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.kind_id=1 AND title.production_year<1999
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id=74343
SELECT COUNT(*) FROM cast_info,movie_companies,title WHERE title.id=movie_companies.movie_id AND title.id=cast_info.movie_id AND title.production_year<1962 AND cast_info.person_id<1419945 AND cast_info.role_id=3
SELECT COUNT(*) FROM movie_companies,movie_info,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info.movie_id AND movie_companies.company_id=11145
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id AND movie_companies.company_id=2440 AND movie_companies.company_type_id=2
SELECT COUNT(*) FROM cast_info,movie_info_idx,title WHERE title.id=cast_info.movie_id AND title.id=movie_info_idx.movie_id AND cast_info.person_id<1163272 AND cast_info.role_id=4
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id>89404
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND title.kind_id>3 AND title.production_year=2012
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id<2697
SELECT COUNT(*) FROM movie_companies,movie_info,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info.movie_id AND movie_companies.company_id=102833 AND movie_companies.company_type_id=2
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.production_year<1996 AND cast_info.person_id<561784 AND cast_info.role_id>2
SELECT COUNT(*) FROM movie_info,movie_info_idx,title WHERE title.id=movie_info.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id<7 AND title.production_year<1984 AND movie_info.info_type_id<16
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.production_year=2012 AND cast_info.role_id=7
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id=18042 AND movie_companies.company_type_id=1
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id<1369270 AND cast_info.role_id>10
SELECT COUNT(*) FROM title WHERE title.kind_id>2 AND title.production_year>2004
SELECT COUNT(*) FROM movie_companies,movie_info_idx,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id=1 AND title.production_year>1971 AND movie_companies.company_id>2586 AND movie_companies.company_type_id=1
SELECT COUNT(*) FROM cast_info,movie_keyword,title WHERE title.id=cast_info.movie_id AND title.id=movie_keyword.movie_id AND title.production_year>2009 AND cast_info.person_id<2337246 AND movie_keyword.keyword_id=8103
SELECT COUNT(*) FROM cast_info,movie_companies,title WHERE title.id=movie_companies.movie_id AND title.id=cast_info.movie_id AND movie_companies.company_id<12386 AND cast_info.person_id=2044876
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id>76844 AND movie_companies.company_type_id>1
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id AND movie_companies.company_id>45697 AND movie_companies.company_type_id=1 AND movie_keyword.keyword_id=2451
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id>2813582 AND cast_info.role_id>4
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id AND title.production_year=1959 AND movie_keyword.keyword_id=9986
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id<12919
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND title.kind_id=7 AND movie_companies.company_id<132783 AND movie_companies.company_type_id<2
SELECT COUNT(*) FROM cast_info,movie_keyword,title WHERE title.id=cast_info.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id>1 AND cast_info.person_id=3962601
SELECT COUNT(*) FROM cast_info,movie_info_idx,title WHERE title.id=cast_info.movie_id AND title.id=movie_info_idx.movie_id AND cast_info.person_id<2845062 AND cast_info.role_id=3 AND movie_info_idx.info_type_id=99
SELECT COUNT(*) FROM cast_info,movie_info,title WHERE title.id=cast_info.movie_id AND title.id=movie_info.movie_id AND cast_info.person_id>540293 AND cast_info.role_id=2
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.kind_id<7 AND cast_info.person_id>2839241 AND cast_info.role_id<10
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.kind_id>1 AND title.production_year>1966 AND cast_info.person_id<3020063
SELECT COUNT(*) FROM movie_info_idx,movie_keyword,title WHERE title.id=movie_info_idx.movie_id AND title.id=movie_keyword.movie_id AND movie_info_idx.info_type_id=101 AND movie_keyword.keyword_id>16308
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id=8631
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id=2890147 AND cast_info.role_id<10
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.production_year<1964 AND movie_info.info_type_id>17
SELECT COUNT(*) FROM cast_info,movie_info,title WHERE title.id=cast_info.movie_id AND title.id=movie_info.movie_id AND title.kind_id=7 AND cast_info.person_id=1759568 AND cast_info.role_id<10
SELECT COUNT(*) FROM movie_companies,movie_info_idx,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info_idx.movie_id AND movie_companies.company_id>554 AND movie_companies.company_type_id<2 AND movie_info_idx.info_type_id=101
SELECT COUNT(*) FROM movie_info,movie_keyword,title WHERE title.id=movie_info.movie_id AND title.id=movie_keyword.movie_id AND movie_keyword.keyword_id<229
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.kind_id=7 AND title.production_year>2000 AND movie_keyword.keyword_id<11769
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.kind_id=7 AND title.production_year>2007 AND movie_keyword.keyword_id<20033
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id<81044
SELECT COUNT(*) FROM movie_info,movie_keyword,title WHERE title.id=movie_info.movie_id AND title.id=movie_keyword.movie_id AND title.production_year=2004 AND movie_info.info_type_id=15
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id<7 AND title.production_year<2003 AND movie_companies.company_id<7596 AND movie_companies.company_type_id>1 AND movie_keyword.keyword_id>137
SELECT COUNT(*) FROM movie_companies,movie_info,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info.movie_id AND title.kind_id<4 AND movie_companies.company_id=11151
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id<18826 AND movie_companies.company_type_id>1
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id>1057
SELECT COUNT(*) FROM movie_info,movie_keyword,title WHERE title.id=movie_info.movie_id AND title.id=movie_keyword.movie_id AND movie_info.info_type_id<5 AND movie_keyword.keyword_id>8190
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id=29028
SELECT COUNT(*) FROM movie_companies,movie_info,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info.movie_id AND movie_companies.company_id<737 AND movie_companies.company_type_id=2 AND movie_info.info_type_id=16
SELECT COUNT(*) FROM movie_companies,movie_info_idx,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info_idx.movie_id AND movie_companies.company_id>43941 AND movie_companies.company_type_id=2 AND movie_info_idx.info_type_id=101
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.kind_id<7 AND title.production_year=2001
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id<5683
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id<2831
SELECT COUNT(*) FROM title WHERE title.production_year=1936
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id=5 AND movie_companies.company_type_id>1
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.production_year=1998 AND movie_info.info_type_id>2
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.production_year<1999 AND movie_keyword.keyword_id=231
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id<137482
SELECT COUNT(*) FROM cast_info,movie_info,title WHERE title.id=cast_info.movie_id AND title.id=movie_info.movie_id AND cast_info.person_id<3476391 AND cast_info.role_id<3 AND movie_info.info_type_id<16
SELECT COUNT(*) FROM movie_info,movie_keyword,title WHERE title.id=movie_info.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id<7 AND movie_info.info_type_id=15 AND movie_keyword.keyword_id=394
SELECT COUNT(*) FROM cast_info,movie_keyword,title WHERE title.id=cast_info.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id=7 AND title.production_year=1979
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id<1801379 AND cast_info.role_id=10
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND title.kind_id=4 AND title.production_year<1991
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.kind_id>1 AND movie_keyword.keyword_id=1878
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id<14545
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id AND movie_companies.company_id=394 AND movie_companies.company_type_id=2
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id=27720 AND movie_companies.company_type_id<2
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id<7 AND title.production_year<2008 AND movie_keyword.keyword_id=245
SELECT COUNT(*) FROM movie_info,movie_keyword,title WHERE title.id=movie_info.movie_id AND title.id=movie_keyword.movie_id AND movie_keyword.keyword_id<6069
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id=11346
SELECT COUNT(*) FROM movie_info,movie_keyword,title WHERE title.id=movie_info.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id=7 AND movie_info.info_type_id=8
SELECT COUNT(*) FROM cast_info,movie_keyword,title WHERE title.id=cast_info.movie_id AND title.id=movie_keyword.movie_id AND title.production_year>2001 AND cast_info.person_id=2926527 AND cast_info.role_id>3
SELECT COUNT(*) FROM movie_info,movie_keyword,title WHERE title.id=movie_info.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id>4 AND movie_info.info_type_id>16
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.kind_id>1 AND title.production_year=2005 AND movie_keyword.keyword_id=5973
SELECT COUNT(*) FROM movie_companies,movie_info_idx,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id>2 AND title.production_year<2005 AND movie_companies.company_id<5395
SELECT COUNT(*) FROM cast_info,movie_companies,title WHERE title.id=movie_companies.movie_id AND title.id=cast_info.movie_id AND cast_info.person_id=1858638 AND cast_info.role_id=2
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND title.kind_id=4 AND title.production_year>1999 AND movie_info_idx.info_type_id=99
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND title.production_year=1972
SELECT COUNT(*) FROM movie_companies,movie_info,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info.movie_id AND title.kind_id=7 AND movie_info.info_type_id=8
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.production_year<1924 AND movie_info.info_type_id<3
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.kind_id=2 AND movie_keyword.keyword_id<4194
SELECT COUNT(*) FROM cast_info,movie_info_idx,title WHERE title.id=cast_info.movie_id AND title.id=movie_info_idx.movie_id AND cast_info.person_id=272695 AND cast_info.role_id<3
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND title.production_year>2011 AND movie_companies.company_id<63857 AND movie_companies.company_type_id<2
SELECT COUNT(*) FROM movie_companies,movie_info,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info.movie_id AND title.kind_id=7 AND title.production_year<2008 AND movie_info.info_type_id<8
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id>1 AND title.production_year>2007 AND movie_keyword.keyword_id=1078
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id<783924 AND cast_info.role_id>4
SELECT COUNT(*) FROM movie_info,movie_keyword,title WHERE title.id=movie_info.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id>3 AND movie_info.info_type_id>8
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id>1413435 AND cast_info.role_id>1
SELECT COUNT(*) FROM movie_info,movie_info_idx,title WHERE title.id=movie_info.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id<6
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id=183597
SELECT COUNT(*) FROM cast_info,movie_info_idx,title WHERE title.id=cast_info.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id=7 AND cast_info.person_id<2285138 AND cast_info.role_id<10
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND cast_info.person_id>1603654 AND cast_info.role_id=1
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.production_year<1944 AND cast_info.role_id>1
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.kind_id=7 AND movie_keyword.keyword_id<35846
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id<2173217 AND cast_info.role_id>4
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND title.production_year<1994 AND movie_companies.company_type_id<2
SELECT COUNT(*) FROM movie_info,movie_info_idx,title WHERE title.id=movie_info.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id=7 AND title.production_year>2012 AND movie_info_idx.info_type_id<100
SELECT COUNT(*) FROM movie_info,movie_keyword,title WHERE title.id=movie_info.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id=2 AND title.production_year>2007 AND movie_info.info_type_id>105
SELECT COUNT(*) FROM cast_info,movie_info,title WHERE title.id=cast_info.movie_id AND title.id=movie_info.movie_id AND cast_info.person_id>2036508 AND cast_info.role_id<8
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id>661
SELECT COUNT(*) FROM movie_info,movie_info_idx,title WHERE title.id=movie_info.movie_id AND title.id=movie_info_idx.movie_id AND title.production_year>1993 AND movie_info_idx.info_type_id=101
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id<2942907 AND cast_info.role_id<10
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id=2017 AND movie_companies.company_type_id<2
SELECT COUNT(*) FROM cast_info,movie_keyword,title WHERE title.id=cast_info.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id<7 AND title.production_year>1994 AND cast_info.person_id<958379 AND cast_info.role_id>1 AND movie_keyword.keyword_id=493
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id>22765
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND title.production_year=2000 AND movie_companies.company_id>53687
SELECT COUNT(*) FROM movie_companies,movie_info,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info.movie_id AND title.production_year<1990
SELECT COUNT(*) FROM movie_info,movie_info_idx,title WHERE title.id=movie_info.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id<7 AND title.production_year<1987
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND title.kind_id<7 AND title.production_year>1918
SELECT COUNT(*) FROM cast_info,movie_info_idx,title WHERE title.id=cast_info.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id>1 AND title.production_year>2012 AND cast_info.person_id<2869422 AND cast_info.role_id<8 AND movie_info_idx.info_type_id>100
SELECT COUNT(*) FROM movie_info,movie_info_idx,title WHERE title.id=movie_info.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id=4 AND title.production_year=1995 AND movie_info.info_type_id<84 AND movie_info_idx.info_type_id<100
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id<474 AND movie_companies.company_type_id=2
SELECT COUNT(*) FROM cast_info,movie_info,title WHERE title.id=cast_info.movie_id AND title.id=movie_info.movie_id AND cast_info.person_id=3156883
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id AND movie_companies.company_id=8651
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.kind_id=4 AND title.production_year>1929 AND movie_info.info_type_id<2
SELECT COUNT(*) FROM cast_info,movie_keyword,title WHERE title.id=cast_info.movie_id AND title.id=movie_keyword.movie_id AND cast_info.person_id=2332542 AND cast_info.role_id>2
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.kind_id<7 AND cast_info.person_id>3205974
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.kind_id=7 AND title.production_year>0 AND movie_keyword.keyword_id<7465
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id=258
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.kind_id=4 AND title.production_year=2000 AND cast_info.person_id>427492 AND cast_info.role_id=1
SELECT COUNT(*) FROM movie_info,movie_info_idx,title WHERE title.id=movie_info.movie_id AND title.id=movie_info_idx.movie_id AND title.production_year<2010 AND movie_info.info_type_id>7 AND movie_info_idx.info_type_id<101
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id<976888
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.kind_id=7 AND movie_info.info_type_id>15
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id=2587
SELECT COUNT(*) FROM title WHERE title.kind_id=2 AND title.production_year=1999
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id=1527
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.kind_id=7 AND title.production_year>2004 AND movie_info.info_type_id<8
SELECT COUNT(*) FROM cast_info,movie_info_idx,title WHERE title.id=cast_info.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id<2 AND title.production_year=2002 AND cast_info.role_id=6 AND movie_info_idx.info_type_id>99
SELECT COUNT(*) FROM movie_info,movie_keyword,title WHERE title.id=movie_info.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id>1 AND movie_info.info_type_id>102
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id=2988 AND movie_companies.company_type_id=1
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND movie_keyword.keyword_id>54183
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id<7 AND title.production_year>2012 AND movie_companies.company_id>40821
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.production_year=2001 AND movie_info.info_type_id>2
SELECT COUNT(*) FROM movie_companies,movie_info_idx,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info_idx.movie_id AND title.production_year>2010 AND movie_info_idx.info_type_id>99
SELECT COUNT(*) FROM cast_info,movie_info,title WHERE title.id=cast_info.movie_id AND title.id=movie_info.movie_id AND title.kind_id=7 AND title.production_year>2006 AND cast_info.person_id<3098461 AND cast_info.role_id=1
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id=15
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.production_year<1986 AND movie_keyword.keyword_id<20324
SELECT COUNT(*) FROM movie_companies,movie_info_idx,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info_idx.movie_id AND movie_companies.company_id=32269 AND movie_companies.company_type_id=2
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND title.kind_id<7 AND title.production_year<2015 AND movie_companies.company_type_id<2
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.kind_id=7 AND title.production_year<1940
SELECT COUNT(*) FROM cast_info,movie_info,title WHERE title.id=cast_info.movie_id AND title.id=movie_info.movie_id AND cast_info.person_id>147146 AND movie_info.info_type_id>4
SELECT COUNT(*) FROM movie_companies,movie_info_idx,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id=7 AND title.production_year=2001 AND movie_info_idx.info_type_id>99
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.production_year=1994
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND movie_keyword.keyword_id=8106
SELECT COUNT(*) FROM movie_info,movie_keyword,title WHERE title.id=movie_info.movie_id AND title.id=movie_keyword.movie_id AND movie_keyword.keyword_id<849
SELECT COUNT(*) FROM movie_companies,movie_info,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info.movie_id AND title.production_year=1959 AND movie_companies.company_type_id=1 AND movie_info.info_type_id<3
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.kind_id=7 AND title.production_year>1932 AND cast_info.role_id>10
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id=7 AND title.production_year=2003 AND movie_companies.company_type_id<2
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id=1095
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id AND movie_companies.company_id=17060 AND movie_keyword.keyword_id=865
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id<3430
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.kind_id<7 AND title.production_year=1923 AND movie_info.info_type_id<16
SELECT COUNT(*) FROM movie_info,movie_info_idx,title WHERE title.id=movie_info.movie_id AND title.id=movie_info_idx.movie_id AND title.production_year<2004 AND movie_info.info_type_id>17
SELECT COUNT(*) FROM cast_info,movie_keyword,title WHERE title.id=cast_info.movie_id AND title.id=movie_keyword.movie_id AND title.production_year>2005 AND cast_info.person_id<2396878
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id>2020 AND movie_companies.company_type_id=1
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id<313397 AND cast_info.role_id=1
SELECT COUNT(*) FROM cast_info,movie_info,title WHERE title.id=cast_info.movie_id AND title.id=movie_info.movie_id AND title.kind_id=3
SELECT COUNT(*) FROM cast_info,movie_info,title WHERE title.id=cast_info.movie_id AND title.id=movie_info.movie_id AND movie_info.info_type_id>98
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND title.kind_id<7 AND title.production_year>1995
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND title.kind_id<7 AND movie_companies.company_id>6
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND title.kind_id<4 AND title.production_year>1965
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id<14853 AND movie_companies.company_type_id>1
SELECT COUNT(*) FROM cast_info,movie_keyword,title WHERE title.id=cast_info.movie_id AND title.id=movie_keyword.movie_id AND title.production_year>2007 AND cast_info.person_id>3497904 AND cast_info.role_id>2
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id=2941252 AND cast_info.role_id=3
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id<2982660
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND movie_keyword.keyword_id>26449
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.production_year>2010 AND cast_info.person_id<49815 AND cast_info.role_id<8
SELECT COUNT(*) FROM movie_info_idx,movie_keyword,title WHERE title.id=movie_info_idx.movie_id AND title.id=movie_keyword.movie_id AND title.production_year=2004 AND movie_info_idx.info_type_id<101 AND movie_keyword.keyword_id=2707
SELECT COUNT(*) FROM cast_info,movie_info_idx,title WHERE title.id=cast_info.movie_id AND title.id=movie_info_idx.movie_id AND cast_info.person_id>3175047 AND cast_info.role_id=10 AND movie_info_idx.info_type_id=100
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND title.kind_id=1 AND title.production_year>1983
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND title.kind_id<7 AND title.production_year<2001
SELECT COUNT(*) FROM movie_companies,movie_info_idx,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info_idx.movie_id AND title.production_year<1966 AND movie_companies.company_type_id=1
SELECT COUNT(*) FROM movie_companies,movie_info_idx,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info_idx.movie_id AND title.production_year>1988 AND movie_info_idx.info_type_id=99
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id<201286
SELECT COUNT(*) FROM movie_companies,movie_info,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info.movie_id AND title.production_year=1998 AND movie_companies.company_id<8684 AND movie_companies.company_type_id>1
SELECT COUNT(*) FROM movie_info,movie_keyword,title WHERE title.id=movie_info.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id<7 AND title.production_year<1990 AND movie_keyword.keyword_id=11934
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id>452
SELECT COUNT(*) FROM movie_companies,movie_info,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info.movie_id AND movie_companies.company_id=20304 AND movie_companies.company_type_id<2 AND movie_info.info_type_id>15
SELECT COUNT(*) FROM cast_info,movie_companies,title WHERE title.id=movie_companies.movie_id AND title.id=cast_info.movie_id AND title.kind_id<7 AND title.production_year>1998
SELECT COUNT(*) FROM movie_info,movie_info_idx,title WHERE title.id=movie_info.movie_id AND title.id=movie_info_idx.movie_id AND title.production_year>1959 AND movie_info.info_type_id=16 AND movie_info_idx.info_type_id=101
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id>10926
SELECT COUNT(*) FROM cast_info,movie_info_idx,title WHERE title.id=cast_info.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id=7 AND title.production_year=2004 AND cast_info.role_id=2 AND movie_info_idx.info_type_id>100
SELECT COUNT(*) FROM movie_info,movie_keyword,title WHERE title.id=movie_info.movie_id AND title.id=movie_keyword.movie_id AND movie_keyword.keyword_id>814
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id=2198
SELECT COUNT(*) FROM movie_info_idx,movie_keyword,title WHERE title.id=movie_info_idx.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id=1 AND title.production_year<1980
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND title.production_year=2007 AND movie_companies.company_id<7838
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id<6404
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND title.kind_id=7 AND title.production_year>2000 AND movie_companies.company_id<21250 AND movie_companies.company_type_id<2
SELECT COUNT(*) FROM cast_info,movie_info,title WHERE title.id=cast_info.movie_id AND title.id=movie_info.movie_id AND title.kind_id=1 AND title.production_year<1982 AND cast_info.role_id>1 AND movie_info.info_type_id<17
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.kind_id<7 AND title.production_year<1991
SELECT COUNT(*) FROM cast_info,movie_keyword,title WHERE title.id=cast_info.movie_id AND title.id=movie_keyword.movie_id AND title.production_year<1991 AND cast_info.person_id>3240646 AND cast_info.role_id>1
SELECT COUNT(*) FROM cast_info,movie_info,title WHERE title.id=cast_info.movie_id AND title.id=movie_info.movie_id AND cast_info.person_id>1979052
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.kind_id=7 AND title.production_year>1950 AND movie_keyword.keyword_id<37100
SELECT COUNT(*) FROM cast_info,movie_info_idx,title WHERE title.id=cast_info.movie_id AND title.id=movie_info_idx.movie_id AND title.production_year<1983
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id<63494 AND movie_companies.company_type_id=2
SELECT COUNT(*) FROM title WHERE title.kind_id>4 AND title.production_year>2003
SELECT COUNT(*) FROM cast_info,movie_info,title WHERE title.id=cast_info.movie_id AND title.id=movie_info.movie_id AND cast_info.person_id>709771 AND cast_info.role_id=8 AND movie_info.info_type_id=1
SELECT COUNT(*) FROM cast_info,movie_info_idx,title WHERE title.id=cast_info.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id=7 AND title.production_year>1955 AND movie_info_idx.info_type_id<100
SELECT COUNT(*) FROM movie_companies,movie_info_idx,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id<7 AND title.production_year>2009 AND movie_info_idx.info_type_id=99
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id>8494
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id>20366 AND movie_companies.company_type_id>1
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id=1077
SELECT COUNT(*) FROM cast_info,movie_keyword,title WHERE title.id=cast_info.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id>4 AND cast_info.person_id>2078207 AND cast_info.role_id>1
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id>1422664 AND cast_info.role_id<9
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id<1076890
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.production_year=2008 AND movie_keyword.keyword_id>1957
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id=4149
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id=1209
SELECT COUNT(*) FROM movie_companies,movie_info,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info.movie_id AND title.kind_id>1 AND movie_companies.company_id>4619 AND movie_companies.company_type_id=1
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.kind_id>1 AND title.production_year>2003 AND movie_keyword.keyword_id<24146
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.kind_id=7 AND title.production_year=2010 AND movie_keyword.keyword_id>5360
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.kind_id<2 AND title.production_year>1989
SELECT COUNT(*) FROM movie_info,movie_info_idx,title WHERE title.id=movie_info.movie_id AND title.id=movie_info_idx.movie_id AND movie_info.info_type_id<3 AND movie_info_idx.info_type_id<100
SELECT COUNT(*) FROM movie_info,movie_info_idx,title WHERE title.id=movie_info.movie_id AND title.id=movie_info_idx.movie_id AND movie_info.info_type_id<15 AND movie_info_idx.info_type_id=101
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id>5537
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.kind_id>1 AND movie_keyword.keyword_id>7766
SELECT COUNT(*) FROM movie_info_idx,movie_keyword,title WHERE title.id=movie_info_idx.movie_id AND title.id=movie_keyword.movie_id AND movie_keyword.keyword_id>5274
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id=9109
SELECT COUNT(*) FROM movie_info_idx,movie_keyword,title WHERE title.id=movie_info_idx.movie_id AND title.id=movie_keyword.movie_id AND movie_info_idx.info_type_id=101 AND movie_keyword.keyword_id<2511
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id>19
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id=1559
SELECT COUNT(*) FROM cast_info,movie_info,title WHERE title.id=cast_info.movie_id AND title.id=movie_info.movie_id AND title.kind_id<7 AND title.production_year=2012 AND movie_info.info_type_id=5
SELECT COUNT(*) FROM cast_info,movie_info_idx,title WHERE title.id=cast_info.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id>1 AND title.production_year>2003 AND movie_info_idx.info_type_id=100
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND movie_keyword.keyword_id<20805
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.kind_id<7 AND title.production_year=2007 AND movie_info.info_type_id=8
SELECT COUNT(*) FROM title WHERE title.kind_id=2 AND title.production_year=1987
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND movie_keyword.keyword_id>36918
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.kind_id=1 AND title.production_year>1947 AND movie_info.info_type_id=5
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.kind_id<7 AND title.production_year>1965
SELECT COUNT(*) FROM cast_info,movie_info_idx,title WHERE title.id=cast_info.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id>3 AND cast_info.person_id<3974927 AND cast_info.role_id<2 AND movie_info_idx.info_type_id=99
SELECT COUNT(*) FROM movie_companies,movie_info_idx,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id=2 AND title.production_year<1975 AND movie_companies.company_type_id=1
SELECT COUNT(*) FROM cast_info,movie_keyword,title WHERE title.id=cast_info.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id=7 AND cast_info.person_id>2825662 AND cast_info.role_id>2
SELECT COUNT(*) FROM cast_info,movie_keyword,title WHERE title.id=cast_info.movie_id AND title.id=movie_keyword.movie_id AND cast_info.person_id=1082588 AND cast_info.role_id<10
SELECT COUNT(*) FROM cast_info,movie_keyword,title WHERE title.id=cast_info.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id=1 AND cast_info.person_id<2761225 AND cast_info.role_id<2 AND movie_keyword.keyword_id<969
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id<37481
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id>1783162 AND cast_info.role_id>10
SELECT COUNT(*) FROM cast_info,movie_companies,title WHERE title.id=movie_companies.movie_id AND title.id=cast_info.movie_id AND movie_companies.company_id>6430
SELECT COUNT(*) FROM title WHERE title.kind_id=1 AND title.production_year<1995
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.production_year<1962 AND cast_info.role_id<3
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id<1769945 AND cast_info.role_id=4
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id<786751 AND cast_info.role_id<10
SELECT COUNT(*) FROM cast_info,movie_companies,title WHERE title.id=movie_companies.movie_id AND title.id=cast_info.movie_id AND movie_companies.company_id>53168 AND movie_companies.company_type_id<2 AND cast_info.person_id<1199928
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id<184
SELECT COUNT(*) FROM cast_info,movie_companies,title WHERE title.id=movie_companies.movie_id AND title.id=cast_info.movie_id AND movie_companies.company_id>107891 AND movie_companies.company_type_id>1 AND cast_info.person_id=938023
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND title.kind_id=1 AND movie_companies.company_id>995
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id<132978 AND cast_info.role_id=9
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.production_year=2012 AND cast_info.role_id<10
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id<900075
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND movie_keyword.keyword_id>23775
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id AND movie_companies.company_id>29178 AND movie_companies.company_type_id>1 AND movie_keyword.keyword_id=17506
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.kind_id>2 AND title.production_year<1999
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND title.production_year<2011 AND movie_companies.company_id>6 AND movie_companies.company_type_id>1
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND cast_info.person_id>152630 AND cast_info.role_id<5
SELECT COUNT(*) FROM cast_info,movie_info_idx,title WHERE title.id=cast_info.movie_id AND title.id=movie_info_idx.movie_id AND cast_info.person_id=2597595 AND cast_info.role_id<8
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND movie_keyword.keyword_id=7771
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.kind_id=7 AND title.production_year<1999 AND movie_info.info_type_id<6
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id<74879 AND movie_companies.company_type_id=2
SELECT COUNT(*) FROM movie_companies,movie_info,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info.movie_id AND title.production_year>2006 AND movie_companies.company_id=1284 AND movie_companies.company_type_id>1 AND movie_info.info_type_id<16
SELECT COUNT(*) FROM title WHERE title.kind_id>2 AND title.production_year>2010
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id>92321
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.production_year>2011
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id>240 AND movie_companies.company_type_id<2
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id=183745 AND movie_companies.company_type_id=2
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND movie_companies.company_id>2165 AND movie_companies.company_type_id<2
SELECT COUNT(*) FROM movie_companies,movie_info,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info.movie_id AND movie_companies.company_id<12121
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.production_year<2005 AND movie_info.info_type_id=16
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id=70124 AND cast_info.role_id>2
SELECT COUNT(*) FROM movie_info,movie_keyword,title WHERE title.id=movie_info.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id=1 AND title.production_year=2010 AND movie_info.info_type_id=18 AND movie_keyword.keyword_id<67902
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND title.production_year>1989 AND movie_companies.company_id<72033
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id>1433320
SELECT COUNT(*) FROM movie_info,movie_keyword,title WHERE title.id=movie_info.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id<7 AND title.production_year=2009 AND movie_keyword.keyword_id<3226
SELECT COUNT(*) FROM cast_info,movie_keyword,title WHERE title.id=cast_info.movie_id AND title.id=movie_keyword.movie_id AND cast_info.person_id<823985
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id=1477844
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.kind_id<7 AND title.production_year<1990
SELECT COUNT(*) FROM cast_info,movie_info_idx,title WHERE title.id=cast_info.movie_id AND title.id=movie_info_idx.movie_id AND title.production_year>2004 AND cast_info.role_id<2
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND title.kind_id<7 AND title.production_year>2001 AND movie_companies.company_id>27 AND movie_companies.company_type_id=2
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND title.kind_id=7 AND title.production_year=1992 AND movie_info_idx.info_type_id>100
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.kind_id=4 AND cast_info.role_id<10
SELECT COUNT(*) FROM movie_info,movie_info_idx,title WHERE title.id=movie_info.movie_id AND title.id=movie_info_idx.movie_id AND title.production_year>2010 AND movie_info.info_type_id=18 AND movie_info_idx.info_type_id<101
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND title.production_year<1908
SELECT COUNT(*) FROM cast_info,movie_info,title WHERE title.id=cast_info.movie_id AND title.id=movie_info.movie_id AND cast_info.person_id<243502 AND cast_info.role_id<3
SELECT COUNT(*) FROM cast_info,movie_info,title WHERE title.id=cast_info.movie_id AND title.id=movie_info.movie_id AND title.production_year<1951
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id>117819
SELECT COUNT(*) FROM cast_info,movie_info_idx,title WHERE title.id=cast_info.movie_id AND title.id=movie_info_idx.movie_id AND cast_info.person_id>1627356 AND movie_info_idx.info_type_id=99
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id=1776501
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id>2173147
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id=2 AND title.production_year<1965 AND movie_companies.company_id<14421 AND movie_companies.company_type_id=2
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.production_year<2004 AND movie_keyword.keyword_id>24382
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id<7851 AND movie_companies.company_type_id=1
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.kind_id=4 AND title.production_year<2000
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.production_year>1914 AND cast_info.person_id<585747 AND cast_info.role_id=10
SELECT COUNT(*) FROM movie_companies,movie_info_idx,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id>1 AND title.production_year<1994
SELECT COUNT(*) FROM movie_info,movie_info_idx,title WHERE title.id=movie_info.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id>1 AND movie_info.info_type_id=1 AND movie_info_idx.info_type_id>100
SELECT COUNT(*) FROM cast_info,movie_companies,title WHERE title.id=movie_companies.movie_id AND title.id=cast_info.movie_id AND title.kind_id<7 AND title.production_year>2009 AND movie_companies.company_id<12808 AND movie_companies.company_type_id=2 AND cast_info.person_id=1740619
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND movie_companies.company_id>22485
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND movie_keyword.keyword_id=31292
SELECT COUNT(*) FROM movie_info_idx,movie_keyword,title WHERE title.id=movie_info_idx.movie_id AND title.id=movie_keyword.movie_id AND movie_keyword.keyword_id<1221
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id>43931
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND title.kind_id<7 AND title.production_year>1977
SELECT COUNT(*) FROM movie_info,movie_info_idx,title WHERE title.id=movie_info.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id<7 AND title.production_year<2003 AND movie_info.info_type_id<98
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.production_year=2001 AND cast_info.person_id<3304257 AND cast_info.role_id>3
SELECT COUNT(*) FROM cast_info,movie_info_idx,title WHERE title.id=cast_info.movie_id AND title.id=movie_info_idx.movie_id AND cast_info.person_id>2771690 AND cast_info.role_id=3
SELECT COUNT(*) FROM cast_info,movie_info_idx,title WHERE title.id=cast_info.movie_id AND title.id=movie_info_idx.movie_id AND title.production_year<2002 AND cast_info.role_id=8
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND title.kind_id=7 AND title.production_year>2003 AND movie_companies.company_id<70905 AND movie_companies.company_type_id=2
SELECT COUNT(*) FROM title WHERE title.kind_id<7 AND title.production_year<1985
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id>761897 AND cast_info.role_id=2
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id>16496
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id AND title.production_year>2012 AND movie_companies.company_id<7404 AND movie_companies.company_type_id=1
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id<11554
SELECT COUNT(*) FROM movie_companies,movie_info_idx,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info_idx.movie_id AND title.production_year>1968 AND movie_companies.company_id<885 AND movie_companies.company_type_id>1
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.production_year>2008 AND movie_info.info_type_id<5
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id>57105
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND title.kind_id=7 AND title.production_year>1979
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id>1687313 AND cast_info.role_id=2
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.production_year>1968 AND movie_info.info_type_id<16
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id>2109427
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.kind_id=7 AND title.production_year<2007 AND movie_keyword.keyword_id<140
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id>11930
SELECT COUNT(*) FROM movie_info,movie_keyword,title WHERE title.id=movie_info.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id=1 AND title.production_year>2009 AND movie_info.info_type_id=4 AND movie_keyword.keyword_id>12675
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.kind_id=7 AND title.production_year=2001 AND movie_info.info_type_id<8
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.kind_id=7 AND cast_info.person_id>373638
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.kind_id<7 AND title.production_year=1975
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id>31889
SELECT COUNT(*) FROM movie_info,movie_info_idx,title WHERE title.id=movie_info.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id=1 AND title.production_year=1995
SELECT COUNT(*) FROM movie_info,movie_info_idx,title WHERE title.id=movie_info.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id=7 AND title.production_year<2000 AND movie_info.info_type_id>98
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND title.production_year=2008 AND movie_companies.company_id>34458 AND movie_companies.company_type_id=1
SELECT COUNT(*) FROM movie_companies,movie_info,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info.movie_id AND title.production_year=2006 AND movie_companies.company_id>23373 AND movie_info.info_type_id=1
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.kind_id=3 AND title.production_year=2005 AND movie_keyword.keyword_id>748
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id=1331371 AND cast_info.role_id>1
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND movie_info.info_type_id>96
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id<9560 AND cast_info.role_id<2
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id<92170
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id AND movie_keyword.keyword_id>7613
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.production_year=1989
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id>105721
SELECT COUNT(*) FROM movie_companies,movie_info,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info.movie_id AND title.kind_id>1 AND title.production_year=2004 AND movie_companies.company_id>9661 AND movie_companies.company_type_id=1 AND movie_info.info_type_id<8
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND title.kind_id<7 AND title.production_year<1989
SELECT COUNT(*) FROM movie_info,movie_keyword,title WHERE title.id=movie_info.movie_id AND title.id=movie_keyword.movie_id AND movie_keyword.keyword_id>12080
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.production_year>2006 AND movie_keyword.keyword_id=2488
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id=2833207 AND cast_info.role_id=3
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id=13670 AND movie_companies.company_type_id=1
SELECT COUNT(*) FROM movie_info_idx,movie_keyword,title WHERE title.id=movie_info_idx.movie_id AND title.id=movie_keyword.movie_id AND title.production_year>2011 AND movie_info_idx.info_type_id<101
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND title.kind_id<4 AND title.production_year<2009 AND movie_info_idx.info_type_id<101
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND movie_companies.company_id>521
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id>33791
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.production_year>2007 AND movie_info.info_type_id<4
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND movie_companies.company_id<51677 AND movie_companies.company_type_id>1
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id=27553
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND title.kind_id<7 AND title.production_year<1989
SELECT COUNT(*) FROM cast_info,movie_info,title WHERE title.id=cast_info.movie_id AND title.id=movie_info.movie_id AND title.production_year<1988 AND cast_info.person_id<1235435 AND cast_info.role_id>1
SELECT COUNT(*) FROM cast_info,movie_companies,title WHERE title.id=movie_companies.movie_id AND title.id=cast_info.movie_id AND movie_companies.company_type_id<2 AND cast_info.person_id<2326570 AND cast_info.role_id=6
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND title.kind_id>3 AND title.production_year>2009 AND movie_companies.company_type_id<2
SELECT COUNT(*) FROM movie_info,movie_keyword,title WHERE title.id=movie_info.movie_id AND title.id=movie_keyword.movie_id AND movie_keyword.keyword_id>2048
SELECT COUNT(*) FROM cast_info,movie_keyword,title WHERE title.id=cast_info.movie_id AND title.id=movie_keyword.movie_id AND cast_info.person_id<2095053 AND movie_keyword.keyword_id=545
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id=25462
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id AND movie_companies.company_id<3393 AND movie_keyword.keyword_id>3068
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id AND movie_companies.company_id>7870 AND movie_keyword.keyword_id>5761
SELECT COUNT(*) FROM movie_info_idx,movie_keyword,title WHERE title.id=movie_info_idx.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id=1 AND title.production_year=1998 AND movie_info_idx.info_type_id>100
SELECT COUNT(*) FROM cast_info,movie_info,title WHERE title.id=cast_info.movie_id AND title.id=movie_info.movie_id AND title.production_year<1995 AND cast_info.role_id=10
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id AND title.production_year=2001 AND movie_keyword.keyword_id>137
SELECT COUNT(*) FROM movie_info,movie_info_idx,title WHERE title.id=movie_info.movie_id AND title.id=movie_info_idx.movie_id AND title.production_year<1975 AND movie_info_idx.info_type_id=100
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.kind_id<7 AND title.production_year=2010 AND movie_info.info_type_id=6
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id=29155 AND movie_companies.company_type_id>1
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id>1407
SELECT COUNT(*) FROM movie_companies,movie_info,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info.movie_id AND title.kind_id>1 AND movie_companies.company_id<14321 AND movie_companies.company_type_id<2 AND movie_info.info_type_id<18
SELECT COUNT(*) FROM cast_info,movie_keyword,title WHERE title.id=cast_info.movie_id AND title.id=movie_keyword.movie_id AND cast_info.person_id<1545964 AND movie_keyword.keyword_id<229
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.kind_id=7 AND cast_info.person_id<1375332
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id>1307112 AND cast_info.role_id=1
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id=1 AND movie_companies.company_id<63237
SELECT COUNT(*) FROM movie_info_idx,movie_keyword,title WHERE title.id=movie_info_idx.movie_id AND title.id=movie_keyword.movie_id AND title.production_year>2006 AND movie_info_idx.info_type_id<100 AND movie_keyword.keyword_id>14219
SELECT COUNT(*) FROM movie_info_idx,movie_keyword,title WHERE title.id=movie_info_idx.movie_id AND title.id=movie_keyword.movie_id AND movie_keyword.keyword_id>6427
SELECT COUNT(*) FROM movie_info WHERE movie_info.info_type_id>35
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id>1238670 AND cast_info.role_id=3
SELECT COUNT(*) FROM cast_info,movie_keyword,title WHERE title.id=cast_info.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id=1 AND movie_keyword.keyword_id>275
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND title.kind_id<7 AND movie_companies.company_id>17292 AND movie_companies.company_type_id>1
SELECT COUNT(*) FROM title WHERE title.production_year=1964
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.kind_id=6 AND title.production_year=1992 AND movie_info.info_type_id>7
SELECT COUNT(*) FROM cast_info,movie_keyword,title WHERE title.id=cast_info.movie_id AND title.id=movie_keyword.movie_id AND title.production_year<1976 AND cast_info.person_id<225468 AND cast_info.role_id=10
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND title.production_year=2005 AND movie_companies.company_id<6 AND movie_companies.company_type_id>1
SELECT COUNT(*) FROM cast_info,movie_info_idx,title WHERE title.id=cast_info.movie_id AND title.id=movie_info_idx.movie_id AND cast_info.person_id<197680 AND movie_info_idx.info_type_id=100
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id=548980 AND cast_info.role_id<10
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id=5886
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id<7 AND title.production_year>1987 AND movie_companies.company_id<35126 AND movie_companies.company_type_id>1
SELECT COUNT(*) FROM movie_companies,movie_info_idx,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id=4 AND title.production_year=2011 AND movie_companies.company_id<16522 AND movie_companies.company_type_id=1 AND movie_info_idx.info_type_id<100
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id>1698666 AND cast_info.role_id=3
SELECT COUNT(*) FROM cast_info,movie_companies,title WHERE title.id=movie_companies.movie_id AND title.id=cast_info.movie_id AND movie_companies.company_type_id=1 AND cast_info.person_id<2806208 AND cast_info.role_id=1
SELECT COUNT(*) FROM cast_info,movie_info_idx,title WHERE title.id=cast_info.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id>1 AND title.production_year=2012
SELECT COUNT(*) FROM cast_info,movie_keyword,title WHERE title.id=cast_info.movie_id AND title.id=movie_keyword.movie_id AND title.production_year<1995 AND cast_info.person_id>2135030 AND movie_keyword.keyword_id>1306
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id AND movie_companies.company_id>30798 AND movie_companies.company_type_id=2
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.kind_id=7 AND title.production_year<2008 AND cast_info.person_id=1186454
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND movie_companies.company_id=24054
SELECT COUNT(*) FROM movie_companies,movie_info,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info.movie_id AND movie_companies.company_id=20634 AND movie_info.info_type_id>16
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id AND title.production_year<2001 AND movie_companies.company_type_id<2 AND movie_keyword.keyword_id=851
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id=7 AND movie_companies.company_type_id=1 AND movie_keyword.keyword_id<34343
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id AND title.production_year>1996 AND movie_keyword.keyword_id<70
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id<43478
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND movie_keyword.keyword_id<28758
SELECT COUNT(*) FROM movie_companies,movie_info_idx,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id=7 AND title.production_year=2011 AND movie_info_idx.info_type_id>100
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id<7 AND title.production_year<1968 AND movie_companies.company_type_id=2 AND movie_keyword.keyword_id>27740
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id<1278087
SELECT COUNT(*) FROM movie_info_idx,movie_keyword,title WHERE title.id=movie_info_idx.movie_id AND title.id=movie_keyword.movie_id AND movie_keyword.keyword_id<18
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id<41003 AND movie_companies.company_type_id=2
SELECT COUNT(*) FROM cast_info,movie_keyword,title WHERE title.id=cast_info.movie_id AND title.id=movie_keyword.movie_id AND movie_keyword.keyword_id<1198
SELECT COUNT(*) FROM cast_info,movie_info_idx,title WHERE title.id=cast_info.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id=3 AND title.production_year>1966 AND cast_info.role_id=9
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.kind_id=1 AND title.production_year>2009
SELECT COUNT(*) FROM cast_info,movie_info,title WHERE title.id=cast_info.movie_id AND title.id=movie_info.movie_id AND title.production_year>1986
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.kind_id=1 AND title.production_year<2007 AND movie_keyword.keyword_id=15005
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND movie_info.info_type_id<90
SELECT COUNT(*) FROM cast_info,movie_keyword,title WHERE title.id=cast_info.movie_id AND title.id=movie_keyword.movie_id AND title.production_year=2007 AND cast_info.role_id>2
SELECT COUNT(*) FROM movie_companies,movie_info,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info.movie_id AND title.kind_id<7 AND title.production_year=1989 AND movie_companies.company_type_id>1
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id<602258 AND cast_info.role_id=1
SELECT COUNT(*) FROM movie_info,movie_info_idx,title WHERE title.id=movie_info.movie_id AND title.id=movie_info_idx.movie_id AND title.production_year>2012 AND movie_info.info_type_id>8 AND movie_info_idx.info_type_id<101
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id AND movie_companies.company_id>233 AND movie_companies.company_type_id>1
SELECT COUNT(*) FROM title WHERE title.kind_id>1 AND title.production_year>1990
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.kind_id=7 AND title.production_year<2003
SELECT COUNT(*) FROM movie_info,movie_info_idx,title WHERE title.id=movie_info.movie_id AND title.id=movie_info_idx.movie_id AND title.production_year>1935 AND movie_info_idx.info_type_id<100
SELECT COUNT(*) FROM movie_info_idx,movie_keyword,title WHERE title.id=movie_info_idx.movie_id AND title.id=movie_keyword.movie_id AND title.production_year>1999 AND movie_info_idx.info_type_id=101 AND movie_keyword.keyword_id<508
SELECT COUNT(*) FROM movie_info,movie_keyword,title WHERE title.id=movie_info.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id<7 AND title.production_year<1926 AND movie_info.info_type_id<5 AND movie_keyword.keyword_id<23822
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id<1050
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.kind_id=7 AND title.production_year<2003 AND movie_keyword.keyword_id=4453
SELECT COUNT(*) FROM movie_companies,movie_info_idx,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id>1 AND movie_companies.company_id=1434
SELECT COUNT(*) FROM cast_info,movie_companies,title WHERE title.id=movie_companies.movie_id AND title.id=cast_info.movie_id AND cast_info.person_id<2070307 AND cast_info.role_id>2
SELECT COUNT(*) FROM cast_info,movie_keyword,title WHERE title.id=cast_info.movie_id AND title.id=movie_keyword.movie_id AND movie_keyword.keyword_id=335
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.kind_id<7 AND movie_keyword.keyword_id<519
SELECT COUNT(*) FROM movie_info,movie_info_idx,title WHERE title.id=movie_info.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id<7 AND title.production_year<1992 AND movie_info_idx.info_type_id>101
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND movie_keyword.keyword_id=16353
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.kind_id>2 AND title.production_year>1989 AND movie_info.info_type_id>16
SELECT COUNT(*) FROM cast_info,movie_companies,title WHERE title.id=movie_companies.movie_id AND title.id=cast_info.movie_id AND title.kind_id=7 AND title.production_year<2006 AND movie_companies.company_id<24741 AND movie_companies.company_type_id>1 AND cast_info.person_id>43656 AND cast_info.role_id>10
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id>13740
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id=3 AND movie_companies.company_type_id=1 AND movie_keyword.keyword_id=3089
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id>543933
SELECT COUNT(*) FROM movie_info_idx,movie_keyword,title WHERE title.id=movie_info_idx.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id>1 AND movie_info_idx.info_type_id=99 AND movie_keyword.keyword_id=8719
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.kind_id>1 AND title.production_year<2007 AND movie_info.info_type_id<98
SELECT COUNT(*) FROM movie_info,movie_info_idx,title WHERE title.id=movie_info.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id>6 AND title.production_year=1954 AND movie_info.info_type_id<7 AND movie_info_idx.info_type_id=99
SELECT COUNT(*) FROM movie_companies,movie_info_idx,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id<7 AND title.production_year>1999 AND movie_companies.company_id>95609
SELECT COUNT(*) FROM movie_companies,movie_info_idx,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info_idx.movie_id AND movie_companies.company_id<11373 AND movie_companies.company_type_id>1 AND movie_info_idx.info_type_id=100
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id=99591 AND movie_companies.company_type_id>1
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.kind_id<4 AND movie_info.info_type_id<106
SELECT COUNT(*) FROM cast_info,movie_info,title WHERE title.id=cast_info.movie_id AND title.id=movie_info.movie_id AND title.kind_id=7 AND title.production_year<1994 AND movie_info.info_type_id=2
SELECT COUNT(*) FROM cast_info,movie_keyword,title WHERE title.id=cast_info.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id=1 AND title.production_year=2007 AND movie_keyword.keyword_id<1060
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id<2252644 AND cast_info.role_id=1
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id>75817
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id=1089450
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id AND movie_companies.company_type_id<2 AND movie_keyword.keyword_id<8816
SELECT COUNT(*) FROM movie_info,movie_keyword,title WHERE title.id=movie_info.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id=7 AND title.production_year<1993 AND movie_keyword.keyword_id>56040
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id<2538293
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.kind_id=7 AND title.production_year=2012 AND movie_keyword.keyword_id>386
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND title.kind_id=7 AND title.production_year=1997 AND movie_info_idx.info_type_id>100
SELECT COUNT(*) FROM movie_companies,movie_info,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info.movie_id AND title.kind_id=7 AND title.production_year=1997 AND movie_companies.company_id>4423 AND movie_companies.company_type_id>1 AND movie_info.info_type_id<4
SELECT COUNT(*) FROM movie_companies,movie_info_idx,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info_idx.movie_id AND title.production_year>1998 AND movie_companies.company_id>16210 AND movie_companies.company_type_id<2 AND movie_info_idx.info_type_id=100
SELECT COUNT(*) FROM cast_info,movie_keyword,title WHERE title.id=cast_info.movie_id AND title.id=movie_keyword.movie_id
SELECT COUNT(*) FROM cast_info,movie_info_idx,title WHERE title.id=cast_info.movie_id AND title.id=movie_info_idx.movie_id AND cast_info.person_id<560899 AND movie_info_idx.info_type_id<100
SELECT COUNT(*) FROM cast_info,movie_info,title WHERE title.id=cast_info.movie_id AND title.id=movie_info.movie_id AND title.kind_id>1 AND movie_info.info_type_id<8
SELECT COUNT(*) FROM cast_info,movie_keyword,title WHERE title.id=cast_info.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id<7 AND title.production_year>2006 AND cast_info.person_id>1725989 AND movie_keyword.keyword_id=490
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id<7 AND title.production_year<1974 AND movie_companies.company_id=19 AND movie_keyword.keyword_id>931
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.production_year>2012 AND movie_info.info_type_id=15
SELECT COUNT(*) FROM cast_info,movie_info_idx,title WHERE title.id=cast_info.movie_id AND title.id=movie_info_idx.movie_id AND title.production_year>1970 AND cast_info.person_id>809890 AND cast_info.role_id>1
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND title.kind_id=7 AND title.production_year>2001 AND movie_companies.company_id=2805
SELECT COUNT(*) FROM movie_info,movie_keyword,title WHERE title.id=movie_info.movie_id AND title.id=movie_keyword.movie_id AND title.production_year=1956
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id<502
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id=2788099
SELECT COUNT(*) FROM movie_companies,movie_info,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info.movie_id AND title.kind_id<7 AND title.production_year<2013 AND movie_companies.company_id=14227 AND movie_companies.company_type_id<2
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.kind_id=7 AND title.production_year<1991 AND cast_info.person_id<3136852
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id<15005
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.kind_id<7 AND title.production_year=2011 AND movie_info.info_type_id>8
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id<11623
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id<168228 AND movie_companies.company_type_id<2
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.production_year>2005 AND cast_info.role_id=11
SELECT COUNT(*) FROM movie_info,movie_info_idx,title WHERE title.id=movie_info.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id=1 AND title.production_year>1944 AND movie_info.info_type_id>3 AND movie_info_idx.info_type_id<101
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND title.kind_id=1 AND title.production_year<1973
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id=1710
SELECT COUNT(*) FROM cast_info,movie_companies,title WHERE title.id=movie_companies.movie_id AND title.id=cast_info.movie_id AND title.production_year>2013 AND cast_info.person_id<3121468 AND cast_info.role_id=4
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id<181163 AND movie_companies.company_type_id=2
SELECT COUNT(*) FROM movie_companies,movie_info,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info.movie_id AND movie_companies.company_id<17386 AND movie_info.info_type_id>16
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND title.production_year>2000 AND movie_companies.company_id<72205 AND movie_companies.company_type_id=1
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id<11717 AND movie_companies.company_type_id=1
SELECT COUNT(*) FROM movie_info,movie_info_idx,title WHERE title.id=movie_info.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id=7 AND title.production_year<1987 AND movie_info_idx.info_type_id<101
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.kind_id<7 AND title.production_year=1975 AND movie_info.info_type_id<94
SELECT COUNT(*) FROM cast_info,movie_keyword,title WHERE title.id=cast_info.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id<7 AND cast_info.person_id>1067634 AND cast_info.role_id>1 AND movie_keyword.keyword_id<3458
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.kind_id=1 AND cast_info.role_id>6
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id<21630
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id<7 AND title.production_year>1992 AND movie_keyword.keyword_id<720
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.kind_id=7 AND title.production_year<1983 AND movie_keyword.keyword_id=317
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND title.kind_id<7 AND movie_companies.company_id>2805 AND movie_companies.company_type_id=2
SELECT COUNT(*) FROM title WHERE title.kind_id>6
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id>8942
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id=459
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id>174536 AND movie_companies.company_type_id>1
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id=72165 AND movie_companies.company_type_id=2
SELECT COUNT(*) FROM title WHERE title.production_year=2004
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id=379
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id<2659790
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.kind_id<7 AND title.production_year=2006 AND movie_keyword.keyword_id=5058
SELECT COUNT(*) FROM movie_companies,movie_info_idx,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info_idx.movie_id AND movie_companies.company_id>474 AND movie_companies.company_type_id<2 AND movie_info_idx.info_type_id>101
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND movie_companies.company_id>20567 AND movie_companies.company_type_id=1
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND title.kind_id<7 AND title.production_year>1973 AND movie_companies.company_id=6
SELECT COUNT(*) FROM cast_info,movie_keyword,title WHERE title.id=cast_info.movie_id AND title.id=movie_keyword.movie_id AND cast_info.role_id<4
SELECT COUNT(*) FROM movie_info,movie_keyword,title WHERE title.id=movie_info.movie_id AND title.id=movie_keyword.movie_id AND movie_info.info_type_id=8 AND movie_keyword.keyword_id<8087
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id<171 AND movie_companies.company_type_id=2
SELECT COUNT(*) FROM movie_info,movie_keyword,title WHERE title.id=movie_info.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id=3 AND movie_info.info_type_id<3
SELECT COUNT(*) FROM cast_info,movie_keyword,title WHERE title.id=cast_info.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id=7 AND title.production_year=2004 AND cast_info.person_id<1820808 AND cast_info.role_id<6 AND movie_keyword.keyword_id=1827
SELECT COUNT(*) FROM cast_info,movie_keyword,title WHERE title.id=cast_info.movie_id AND title.id=movie_keyword.movie_id AND cast_info.role_id=10 AND movie_keyword.keyword_id=1748
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id>11146
SELECT COUNT(*) FROM cast_info,movie_info_idx,title WHERE title.id=cast_info.movie_id AND title.id=movie_info_idx.movie_id AND title.production_year<2001 AND cast_info.person_id=2900255 AND cast_info.role_id>1
SELECT COUNT(*) FROM movie_info,movie_keyword,title WHERE title.id=movie_info.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id=7 AND title.production_year>1985 AND movie_keyword.keyword_id=7319
SELECT COUNT(*) FROM movie_info,movie_keyword,title WHERE title.id=movie_info.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id=7 AND movie_info.info_type_id<8 AND movie_keyword.keyword_id=46745
SELECT COUNT(*) FROM cast_info,movie_info,title WHERE title.id=cast_info.movie_id AND title.id=movie_info.movie_id AND title.kind_id>3 AND title.production_year<2011 AND cast_info.person_id<2006264 AND cast_info.role_id<6
SELECT COUNT(*) FROM movie_info_idx,movie_keyword,title WHERE title.id=movie_info_idx.movie_id AND title.id=movie_keyword.movie_id AND title.production_year=2009 AND movie_keyword.keyword_id<1951
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id<79655 AND movie_companies.company_type_id=2
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND title.kind_id=7 AND title.production_year>1912 AND movie_companies.company_type_id=2
SELECT COUNT(*) FROM cast_info,movie_keyword,title WHERE title.id=cast_info.movie_id AND title.id=movie_keyword.movie_id AND title.production_year=1943 AND cast_info.person_id>1367592
SELECT COUNT(*) FROM cast_info,movie_companies,title WHERE title.id=movie_companies.movie_id AND title.id=cast_info.movie_id AND title.production_year>2010 AND movie_companies.company_id<6101 AND movie_companies.company_type_id>1 AND cast_info.person_id<451486 AND cast_info.role_id>1
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id<21864
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.kind_id=1 AND title.production_year<1997
SELECT COUNT(*) FROM movie_companies,movie_info_idx,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id>1 AND movie_companies.company_type_id<2 AND movie_info_idx.info_type_id=101
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id=1332787
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id=4044796 AND cast_info.role_id>2
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id=14238
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.kind_id=7 AND cast_info.person_id>1500621 AND cast_info.role_id>3
SELECT COUNT(*) FROM cast_info,movie_companies,title WHERE title.id=movie_companies.movie_id AND title.id=cast_info.movie_id AND title.production_year=1974 AND movie_companies.company_type_id=2 AND cast_info.role_id>10
SELECT COUNT(*) FROM movie_companies,movie_info,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info.movie_id AND title.kind_id<7 AND title.production_year<2003 AND movie_companies.company_id<89028
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND title.kind_id<7 AND movie_companies.company_id>125
SELECT COUNT(*) FROM movie_companies,movie_info_idx,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info_idx.movie_id AND title.production_year=1998 AND movie_info_idx.info_type_id=99
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id>5946
SELECT COUNT(*) FROM cast_info,movie_keyword,title WHERE title.id=cast_info.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id=7 AND cast_info.person_id>3061647 AND cast_info.role_id>1 AND movie_keyword.keyword_id<7692
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND title.kind_id<7 AND movie_companies.company_id>128
SELECT COUNT(*) FROM cast_info,movie_companies,title WHERE title.id=movie_companies.movie_id AND title.id=cast_info.movie_id AND title.kind_id<4 AND title.production_year<2009
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id>2085
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND title.kind_id=3 AND title.production_year<2006 AND movie_info_idx.info_type_id>100
SELECT COUNT(*) FROM title WHERE title.kind_id=7 AND title.production_year=1978
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.production_year=2009 AND movie_keyword.keyword_id>816
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id<11125
SELECT COUNT(*) FROM movie_info,movie_keyword,title WHERE title.id=movie_info.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id=7 AND title.production_year>2010 AND movie_keyword.keyword_id>889
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id=23863 AND movie_companies.company_type_id=1
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND title.kind_id>1 AND title.production_year=2002 AND movie_info_idx.info_type_id<101
SELECT COUNT(*) FROM movie_info_idx,movie_keyword,title WHERE title.id=movie_info_idx.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id>1 AND title.production_year<2000 AND movie_keyword.keyword_id>8533
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.kind_id=1 AND title.production_year<2013 AND cast_info.role_id=10
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id<7 AND title.production_year=1978 AND movie_keyword.keyword_id<2443
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND cast_info.person_id<837462
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id=5455 AND movie_companies.company_type_id=2
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.kind_id<3 AND cast_info.person_id<2993200
SELECT COUNT(*) FROM cast_info,movie_keyword,title WHERE title.id=cast_info.movie_id AND title.id=movie_keyword.movie_id AND title.production_year<1981 AND movie_keyword.keyword_id<8193
SELECT COUNT(*) FROM cast_info,movie_info,title WHERE title.id=cast_info.movie_id AND title.id=movie_info.movie_id AND title.production_year=2008 AND cast_info.role_id>1 AND movie_info.info_type_id>4
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.kind_id>4 AND title.production_year=2007
SELECT COUNT(*) FROM movie_companies,movie_info,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info.movie_id AND movie_companies.company_id=85166
SELECT COUNT(*) FROM movie_info,movie_info_idx,title WHERE title.id=movie_info.movie_id AND title.id=movie_info_idx.movie_id AND title.production_year=2009 AND movie_info.info_type_id=2 AND movie_info_idx.info_type_id=101
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id>10393
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.kind_id=2 AND movie_keyword.keyword_id>16302
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.kind_id<2 AND title.production_year=2010 AND movie_info.info_type_id>1
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND title.kind_id>1 AND movie_companies.company_id=2805 AND movie_companies.company_type_id>1
SELECT COUNT(*) FROM movie_info,movie_keyword,title WHERE title.id=movie_info.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id>1 AND title.production_year>2012
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.production_year>2004 AND movie_keyword.keyword_id<3505
SELECT COUNT(*) FROM cast_info,movie_keyword,title WHERE title.id=cast_info.movie_id AND title.id=movie_keyword.movie_id AND movie_keyword.keyword_id>7850
SELECT COUNT(*) FROM cast_info,movie_keyword,title WHERE title.id=cast_info.movie_id AND title.id=movie_keyword.movie_id AND title.production_year=2001 AND cast_info.person_id<3012540 AND cast_info.role_id>2 AND movie_keyword.keyword_id>2926
SELECT COUNT(*) FROM movie_info,movie_info_idx,title WHERE title.id=movie_info.movie_id AND title.id=movie_info_idx.movie_id AND movie_info.info_type_id=18 AND movie_info_idx.info_type_id>101
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND movie_companies.company_id<166 AND movie_companies.company_type_id<2
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id=12644 AND movie_companies.company_type_id=2
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.production_year<1983 AND movie_keyword.keyword_id<1198
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.kind_id=7 AND title.production_year<2009 AND movie_keyword.keyword_id>4605
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.kind_id=7 AND title.production_year>1909 AND cast_info.role_id>10
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id<7 AND movie_companies.company_id=15184 AND movie_keyword.keyword_id>1084
SELECT COUNT(*) FROM cast_info,movie_keyword,title WHERE title.id=cast_info.movie_id AND title.id=movie_keyword.movie_id AND movie_keyword.keyword_id<1714
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id<15624
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id=211126 AND movie_companies.company_type_id=2
SELECT COUNT(*) FROM movie_companies,movie_info,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info.movie_id AND title.production_year<2010 AND movie_companies.company_id<12301 AND movie_companies.company_type_id<2
SELECT COUNT(*) FROM movie_info,movie_info_idx,title WHERE title.id=movie_info.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id<7 AND title.production_year<1978 AND movie_info.info_type_id>3
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id=909660
SELECT COUNT(*) FROM cast_info,movie_info,title WHERE title.id=cast_info.movie_id AND title.id=movie_info.movie_id AND title.kind_id=7 AND title.production_year=2009 AND movie_info.info_type_id<8
SELECT COUNT(*) FROM movie_companies,movie_info_idx,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info_idx.movie_id AND title.production_year<1998 AND movie_info_idx.info_type_id=101
SELECT COUNT(*) FROM movie_info,movie_keyword,title WHERE title.id=movie_info.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id<7 AND title.production_year<2000 AND movie_keyword.keyword_id>228
SELECT COUNT(*) FROM movie_info,movie_keyword,title WHERE title.id=movie_info.movie_id AND title.id=movie_keyword.movie_id AND title.production_year<2008 AND movie_info.info_type_id>16
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id>12959 AND movie_companies.company_type_id=1
SELECT COUNT(*) FROM cast_info,movie_companies,title WHERE title.id=movie_companies.movie_id AND title.id=cast_info.movie_id AND title.kind_id<2 AND title.production_year>1987
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND title.kind_id<7 AND title.production_year<1972 AND movie_info_idx.info_type_id=99
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.kind_id<4 AND title.production_year<1992 AND cast_info.person_id<1459803 AND cast_info.role_id=2
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id=84697 AND movie_companies.company_type_id>1
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id<3055
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND cast_info.person_id<1935414 AND cast_info.role_id<3
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id=4023831 AND cast_info.role_id=10
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND movie_companies.company_id>664
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.production_year<1967 AND cast_info.person_id>1178013
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id=46 AND movie_companies.company_type_id=2
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id>14225
SELECT COUNT(*) FROM cast_info,movie_info,title WHERE title.id=cast_info.movie_id AND title.id=movie_info.movie_id AND title.kind_id<7 AND title.production_year=2009 AND cast_info.person_id<1819945 AND cast_info.role_id>2 AND movie_info.info_type_id<17
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.kind_id>3 AND title.production_year<2003
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.kind_id<7 AND title.production_year<2002 AND movie_keyword.keyword_id=335
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND cast_info.person_id<2757041 AND cast_info.role_id<4
SELECT COUNT(*) FROM movie_info,movie_keyword,title WHERE title.id=movie_info.movie_id AND title.id=movie_keyword.movie_id AND movie_keyword.keyword_id=9837
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.kind_id=7 AND title.production_year=2007 AND movie_info.info_type_id=15
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id>1 AND title.production_year>1990 AND movie_companies.company_id>14002 AND movie_companies.company_type_id=2
SELECT COUNT(*) FROM cast_info,movie_companies,title WHERE title.id=movie_companies.movie_id AND title.id=cast_info.movie_id AND movie_companies.company_type_id=2 AND cast_info.person_id<1381413
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id<1116
SELECT COUNT(*) FROM movie_companies,movie_info_idx,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info_idx.movie_id AND movie_companies.company_id<83236 AND movie_companies.company_type_id<2 AND movie_info_idx.info_type_id=100
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND title.production_year>2005 AND movie_companies.company_id<71614 AND movie_companies.company_type_id<2
SELECT COUNT(*) FROM movie_companies,movie_info_idx,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id<7 AND title.production_year<1962 AND movie_info_idx.info_type_id>101
SELECT COUNT(*) FROM movie_info_idx,movie_keyword,title WHERE title.id=movie_info_idx.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id<7 AND title.production_year>2002 AND movie_keyword.keyword_id=3607
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND movie_companies.company_id<67 AND movie_companies.company_type_id<2
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND title.kind_id>1 AND title.production_year<1986 AND movie_companies.company_id<13251 AND movie_companies.company_type_id<2
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND cast_info.person_id=4047986
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id>11151 AND movie_companies.company_type_id<2
SELECT COUNT(*) FROM movie_info,movie_info_idx,title WHERE title.id=movie_info.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id<2 AND movie_info.info_type_id>8
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id>604216
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id=715265
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.kind_id=7 AND movie_keyword.keyword_id>356
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id<335
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND title.kind_id<4 AND title.production_year>1991 AND movie_companies.company_id=6 AND movie_companies.company_type_id<2
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.production_year>2007
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id>22322
SELECT COUNT(*) FROM movie_info,movie_info_idx,title WHERE title.id=movie_info.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id=7 AND movie_info.info_type_id<8 AND movie_info_idx.info_type_id=100
SELECT COUNT(*) FROM movie_info,movie_info_idx,title WHERE title.id=movie_info.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id<7 AND movie_info_idx.info_type_id>100
SELECT COUNT(*) FROM movie_info_idx,movie_keyword,title WHERE title.id=movie_info_idx.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id=1 AND movie_info_idx.info_type_id<100 AND movie_keyword.keyword_id<72503
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id>37608
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.kind_id>1 AND title.production_year=2004 AND cast_info.person_id=1109161 AND cast_info.role_id<4
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id=769
SELECT COUNT(*) FROM movie_info_idx,movie_keyword,title WHERE title.id=movie_info_idx.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id=1 AND title.production_year<2012 AND movie_info_idx.info_type_id<100 AND movie_keyword.keyword_id<3875
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.kind_id>1 AND title.production_year=1932 AND movie_info.info_type_id<3
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id AND title.production_year<2012 AND movie_companies.company_id>17497 AND movie_keyword.keyword_id<38908
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND title.production_year>2012 AND movie_companies.company_id>11961
SELECT COUNT(*) FROM movie_info,movie_keyword,title WHERE title.id=movie_info.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id=7 AND title.production_year>1983 AND movie_info.info_type_id<4
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.kind_id=7 AND title.production_year<2006 AND cast_info.person_id=2749275 AND cast_info.role_id>2
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.kind_id=7 AND title.production_year>1954 AND movie_keyword.keyword_id>5978
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND title.kind_id=4 AND title.production_year>1955
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.kind_id=7 AND cast_info.person_id>2565865 AND cast_info.role_id<5
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id>2963
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id<45686
SELECT COUNT(*) FROM movie_keyword WHERE movie_keyword.keyword_id>2853
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id>809208 AND cast_info.role_id<2
SELECT COUNT(*) FROM cast_info,movie_info_idx,title WHERE title.id=cast_info.movie_id AND title.id=movie_info_idx.movie_id AND cast_info.role_id=8 AND movie_info_idx.info_type_id>99
SELECT COUNT(*) FROM cast_info,movie_companies,title WHERE title.id=movie_companies.movie_id AND title.id=cast_info.movie_id AND movie_companies.company_id<32 AND cast_info.person_id<587722
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id<18958 AND movie_companies.company_type_id<2
SELECT COUNT(*) FROM cast_info,movie_keyword,title WHERE title.id=cast_info.movie_id AND title.id=movie_keyword.movie_id AND title.production_year<2007 AND movie_keyword.keyword_id=2862
SELECT COUNT(*) FROM movie_info_idx,title WHERE title.id=movie_info_idx.movie_id AND title.kind_id<7 AND title.production_year>0 AND movie_info_idx.info_type_id<100
SELECT COUNT(*) FROM movie_companies,movie_info,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info.movie_id AND title.kind_id<7 AND movie_companies.company_id<8185 AND movie_companies.company_type_id<2
SELECT COUNT(*) FROM cast_info,movie_info_idx,title WHERE title.id=cast_info.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id=1 AND title.production_year>2007 AND cast_info.person_id>2577806 AND cast_info.role_id>2
SELECT COUNT(*) FROM movie_info_idx,movie_keyword,title WHERE title.id=movie_info_idx.movie_id AND title.id=movie_keyword.movie_id AND movie_info_idx.info_type_id>101 AND movie_keyword.keyword_id>6043
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id>10583 AND movie_companies.company_type_id=1
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.production_year<1995 AND movie_info.info_type_id=8
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.kind_id=4 AND cast_info.role_id>2
SELECT COUNT(*) FROM movie_info,movie_keyword,title WHERE title.id=movie_info.movie_id AND title.id=movie_keyword.movie_id AND title.production_year>2009 AND movie_info.info_type_id=17 AND movie_keyword.keyword_id>3401
SELECT COUNT(*) FROM cast_info,movie_keyword,title WHERE title.id=cast_info.movie_id AND title.id=movie_keyword.movie_id AND title.production_year>1956
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id=3103961
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id AND title.production_year>2011 AND movie_companies.company_type_id<2 AND movie_keyword.keyword_id<5529
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id>13256 AND movie_companies.company_type_id>1
SELECT COUNT(*) FROM movie_info,movie_info_idx,title WHERE title.id=movie_info.movie_id AND title.id=movie_info_idx.movie_id AND title.production_year>2001 AND movie_info.info_type_id>7
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id>1473822
SELECT COUNT(*) FROM movie_info,movie_info_idx,title WHERE title.id=movie_info.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id<7 AND title.production_year>1995
SELECT COUNT(*) FROM movie_info,movie_info_idx,title WHERE title.id=movie_info.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id=7 AND title.production_year=2004 AND movie_info.info_type_id<16
SELECT COUNT(*) FROM cast_info,movie_info_idx,title WHERE title.id=cast_info.movie_id AND title.id=movie_info_idx.movie_id AND title.kind_id=7 AND title.production_year=2010 AND cast_info.role_id>1
SELECT COUNT(*) FROM title WHERE title.kind_id=7 AND title.production_year>2004
SELECT COUNT(*) FROM cast_info,movie_info,title WHERE title.id=cast_info.movie_id AND title.id=movie_info.movie_id AND title.kind_id=7 AND cast_info.role_id=2 AND movie_info.info_type_id=104
SELECT COUNT(*) FROM movie_keyword,title WHERE title.id=movie_keyword.movie_id AND title.production_year<1983
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND title.kind_id<7 AND title.production_year=2011 AND movie_companies.company_id>14145 AND movie_companies.company_type_id=2
SELECT COUNT(*) FROM movie_info,movie_keyword,title WHERE title.id=movie_info.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id<7 AND movie_info.info_type_id=2 AND movie_keyword.keyword_id=20170
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id AND title.production_year>2011 AND movie_companies.company_type_id=1
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.kind_id=7 AND title.production_year<2009 AND movie_info.info_type_id<4
SELECT COUNT(*) FROM cast_info WHERE cast_info.person_id<2111228 AND cast_info.role_id>10
SELECT COUNT(*) FROM cast_info,movie_companies,title WHERE title.id=movie_companies.movie_id AND title.id=cast_info.movie_id AND title.kind_id>3 AND title.production_year<1955 AND cast_info.role_id<5
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND title.kind_id>1 AND title.production_year<1988 AND movie_companies.company_type_id>1
SELECT COUNT(*) FROM cast_info,title WHERE title.id=cast_info.movie_id AND title.kind_id>1 AND title.production_year=2007 AND cast_info.role_id=1
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.kind_id<7 AND title.production_year=1969
SELECT COUNT(*) FROM movie_companies,movie_keyword,title WHERE title.id=movie_companies.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id=1 AND movie_keyword.keyword_id>22371
SELECT COUNT(*) FROM movie_info,movie_keyword,title WHERE title.id=movie_info.movie_id AND title.id=movie_keyword.movie_id AND movie_keyword.keyword_id<21920
SELECT COUNT(*) FROM cast_info,movie_info,title WHERE title.id=cast_info.movie_id AND title.id=movie_info.movie_id AND movie_info.info_type_id<13
SELECT COUNT(*) FROM cast_info,movie_info,title WHERE title.id=cast_info.movie_id AND title.id=movie_info.movie_id AND title.kind_id=1 AND title.production_year>2002 AND cast_info.person_id<925423 AND cast_info.role_id>1 AND movie_info.info_type_id=7
SELECT COUNT(*) FROM movie_companies,movie_info_idx,title WHERE title.id=movie_companies.movie_id AND title.id=movie_info_idx.movie_id AND title.production_year>1948 AND movie_companies.company_id>5376 AND movie_companies.company_type_id=1
SELECT COUNT(*) FROM movie_companies WHERE movie_companies.company_id<140280 AND movie_companies.company_type_id=1
SELECT COUNT(*) FROM movie_companies,title WHERE title.id=movie_companies.movie_id AND title.kind_id=7 AND movie_companies.company_id<168444
SELECT COUNT(*) FROM title WHERE title.kind_id=1 AND title.production_year>2011
SELECT COUNT(*) FROM cast_info,movie_keyword,title WHERE title.id=cast_info.movie_id AND title.id=movie_keyword.movie_id AND title.kind_id>4 AND cast_info.person_id=1909685 AND movie_keyword.keyword_id<8080
SELECT COUNT(*) FROM movie_info,title WHERE title.id=movie_info.movie_id AND title.kind_id>1 AND title.production_year>2002 AND movie_info.info_type_id<105
