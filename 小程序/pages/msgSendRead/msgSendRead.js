

const app = getApp()

Page({
  data:{

    productId: app.globalData.productId,
    deviceName: app.globalData.deviceName,

    userInfo:{},
    sliderValue:20
  },


  onLoad() {
    
    /*获取用户信息 */
    var that = this
    // wx.getStorage({
    //   key: "userInfo",
    // }).then(res => {
    //   that.setData({
    //     userInfo: JSON.parse(res.data)
    //   })
    // })

    
  },

  toIndividualPage(event){
    let person=event.currentTarget.dataset.person;
    console.log(person._openid)
    wx.navigateTo({
      url:'/pages/individualPage/individualPage?personId=' + person._openid
    })
  },

  bindChanging(e){
    this.setData({
      sliderValue:e.detail.value
    })
  },

  switchChange(e) {
    let value = 0
    if (e.detail.value == true) {
      value = 1
    }
    let item = e.currentTarget.dataset.item
 
    let obj = {
      [`${item}`]: value
    }
    let payload = JSON.stringify(obj)
    JSON.parse
    console.log(payload)
    wx.showLoading()
    wx.cloud.callFunction({
      name: 'iothub-publish',
      data: {
        SecretId: app.globalData.secretId,
        SecretKey: app.globalData.secretKey,
        ProductId: app.globalData.productId,
        DeviceName: app.globalData.deviceName,
        Topic: app.globalData.productId + "/" + app.globalData.deviceName + "/data",
        Payload: payload,
      },
      success: res => {
        wx.showToast({
          icon: 'none',
          title: 'publish完成',
        })
        console.log("res:", res)
      },
      fail: err => {
        wx.showToast({
          icon: 'none',
          title: 'publish失败，请连接设备',
        })
        console.error('[云函数] [iotexplorer] 调用失败：', err)
      }
    })  
  },

  unloadPage:function(){

  },
  /**
   * 生命周期函数--监听页面初次渲染完成
   */
  onReady: function () {

  },

  /**
   * 生命周期函数--监听页面显示
   */
  onShow: function () {

  },

  onUnload:function() {
    this.unloadPage();
    // udp.close();//退出页面时将socket关闭
  }
})


