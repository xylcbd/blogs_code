# -*- coding: UTF-8 -*-
import scrapy
from scrapy.spiders import CrawlSpider
from scrapy.selector import Selector
from scrapy.http import Request
from baidu_spider.items import BaiduSpiderItem
from selenium import webdriver
import time

class BaiduSpider(CrawlSpider):
    name='baidu_spider'
    
    def start_requests(self):
        #函数名不可更改，此函数定义了爬虫的入口地址
        #使用浏览器访问
        self.browser = webdriver.Chrome('d:/software/chromedriver.exe')
        for i in range(0,20,10):
            url = 'https://www.baidu.com/s?wd=爬虫&pn=%d' % i
            yield self.make_requests_from_url(url)

    def parse(self, response):
        #函数名不可更改，此函数定义了爬虫的页面解析
        #打开浏览器访问页面
        self.browser.get(response.url)
        time.sleep(1)
        selector = Selector(text=self.browser.page_source)
        #selector = Selector(response)
        page_start = int(response.url.split('=')[-1])
        
        for i in range(1,11):
            item = BaiduSpiderItem()        
            xpath = '//*[@id="%d"]/h3/a' % (page_start+i)            
            item['url'] = selector.xpath(xpath + '/@href').extract()[0]           
            item['title'] = selector.xpath(xpath + '//text()').extract()           
            yield item

