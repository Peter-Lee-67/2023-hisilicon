//app.js
App({
  globalData: {
    productId: "BO3ULVVKRY", // 产品ID
    deviceName: "Hi3861_mqtt", // 设备名称

    secretId: "AKIDdEdtvoVyDE1e4II2ZrtfFZgW6QUItEJ9",
    secretKey: "c1B6lDOM5ZnlW3NRIwQhlbDfx2tWwCF4",
  },

  onLaunch: function () {
    if (!wx.cloud) {
      console.error('请使用 2.2.3 或以上的基础库以使用云能力')
    } else {
      wx.cloud.init({
        env: "mqtt-8gxveyk8f6947eec",
        traceUser: true,
      })
    }

 
   
    try {
      var value = wx.getStorageSync('userInfo')
      if (value) {
        this.globalData.userInfo = JSON.parse(value);
      }
    } catch (e) {
      console.log('app js:', '用户未登录')
    }
  }


})