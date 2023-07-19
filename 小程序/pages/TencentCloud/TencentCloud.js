const app = getApp()

Page({
  data: {
    productId: app.globalData.productId,
    deviceName: app.globalData.deviceName,
    stateReported: {},
    Result:"无",
    userInfo:{}
  }, 
  onLoad: function (options) {

   /*获取用户信息 */
   var that = this
  //  wx.getStorage({
  //    key: "userInfo",
  //  }).then(res => {
  //    that.setData({
  //      userInfo: JSON.parse(res.data)
  //    })
  //  })

  //   console.log("index onLoad")
    if (!app.globalData.productId) {
      wx.showToast({
        title: "产品ID不能为空",
        icon: 'none',
        duration: 3000
      })
      return
    } else if (!app.globalData.deviceName) {
      wx.showToast({
        title: "设备名称不能为空",
        icon: 'none',
        duration: 3000
      })
      return
    }

    // this.update()
  },
  

  update() {
    wx.showLoading()
    wx.cloud.callFunction({
      name: 'iothub-shadow-query',
      data: {
        ProductId: app.globalData.productId,
        DeviceName: app.globalData.deviceName,
        SecretId: app.globalData.secretId,
        SecretKey: app.globalData.secretKey,
      },
      success: res => {
        wx.showToast({
          icon: 'none',
          title: 'Subscribe完成，获取云端数据成功',
        })
        let deviceData = JSON.parse(res.result.Data)

        this.setData({
          stateReported: deviceData.payload.state.reported
        })
        
        if(this.data.stateReported.temperature == -97){
          this.setData({
            Result: "纵向裂纹"
          })
        }
        if(this.data.stateReported.temperature == -95){
          this.setData({
            Result: "横向裂纹"
          })
        }
        console.log(this.data.Result)
        console.log("result:", deviceData)
      },
      fail: err => {
        wx.showToast({
          icon: 'none',
          title: 'Subscribe失败，获取云端数据失败',
        })
        console.error('[云函数] [iotexplorer] 调用失败：', err)
      }
    })
  },

  toIndividualPage(event){
    let person=event.currentTarget.dataset.person;
    console.log(person._openid)
    wx.navigateTo({
      url:'/pages/individualPage/individualPage?personId=' + person._openid
    })
  },

    /**
   * 生命周期函数--监听页面显示
   */
  onShow: function () {
  
  },
  /**
   * 生命周期函数--监听页面卸载
   */
  onUnload: function () {

  },
    /**
   * 页面相关事件处理函数--监听用户下拉动作
   */
  onPullDownRefresh: function () {

  },
  
  /**
   * 页面上拉触底事件的处理函数
   */
  onReachBottom: function () {

  },
})
