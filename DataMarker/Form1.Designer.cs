namespace DataMarker
{
    partial class Form1
    {
        /// <summary>
        /// 必需的设计器变量。
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// 清理所有正在使用的资源。
        /// </summary>
        /// <param name="disposing">如果应释放托管资源，为 true；否则为 false。</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows 窗体设计器生成的代码

        /// <summary>
        /// 设计器支持所需的方法 - 不要修改
        /// 使用代码编辑器修改此方法的内容。
        /// </summary>
        private void InitializeComponent()
        {
            this.prev_img = new System.Windows.Forms.Button();
            this.save_list = new System.Windows.Forms.Button();
            this.next_img = new System.Windows.Forms.Button();
            this.mark_pass = new System.Windows.Forms.Button();
            this.load_list = new System.Windows.Forms.Button();
            this.mark_fail = new System.Windows.Forms.Button();
            this.tableLayoutPanel1 = new System.Windows.Forms.TableLayoutPanel();
            this.ImageBox = new System.Windows.Forms.PictureBox();
            this.tableLayoutPanel1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.ImageBox)).BeginInit();
            this.SuspendLayout();
            // 
            // prev_img
            // 
            this.prev_img.AutoSize = true;
            this.prev_img.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.prev_img.Dock = System.Windows.Forms.DockStyle.Fill;
            this.prev_img.Location = new System.Drawing.Point(3, 3);
            this.prev_img.Name = "prev_img";
            this.prev_img.Size = new System.Drawing.Size(321, 68);
            this.prev_img.TabIndex = 1;
            this.prev_img.Text = "👈";
            this.prev_img.UseVisualStyleBackColor = true;
            this.prev_img.Click += new System.EventHandler(this.prev_img_Click);
            // 
            // save_list
            // 
            this.save_list.AutoSize = true;
            this.save_list.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.save_list.Dock = System.Windows.Forms.DockStyle.Fill;
            this.save_list.Location = new System.Drawing.Point(330, 3);
            this.save_list.Name = "save_list";
            this.save_list.Size = new System.Drawing.Size(321, 68);
            this.save_list.TabIndex = 2;
            this.save_list.Text = "💾";
            this.save_list.UseVisualStyleBackColor = true;
            // 
            // next_img
            // 
            this.next_img.AutoSize = true;
            this.next_img.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.next_img.Dock = System.Windows.Forms.DockStyle.Fill;
            this.next_img.Location = new System.Drawing.Point(657, 3);
            this.next_img.Name = "next_img";
            this.next_img.Size = new System.Drawing.Size(323, 68);
            this.next_img.TabIndex = 3;
            this.next_img.Text = "👉";
            this.next_img.UseVisualStyleBackColor = true;
            this.next_img.Click += new System.EventHandler(this.next_img_Click);
            // 
            // mark_pass
            // 
            this.mark_pass.AutoSize = true;
            this.mark_pass.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.mark_pass.Dock = System.Windows.Forms.DockStyle.Fill;
            this.mark_pass.Location = new System.Drawing.Point(3, 77);
            this.mark_pass.Name = "mark_pass";
            this.mark_pass.Size = new System.Drawing.Size(321, 68);
            this.mark_pass.TabIndex = 4;
            this.mark_pass.Text = "√";
            this.mark_pass.UseVisualStyleBackColor = true;
            this.mark_pass.Click += new System.EventHandler(this.mark_pass_Click);
            // 
            // load_list
            // 
            this.load_list.AutoSize = true;
            this.load_list.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.load_list.Dock = System.Windows.Forms.DockStyle.Fill;
            this.load_list.Location = new System.Drawing.Point(330, 77);
            this.load_list.Name = "load_list";
            this.load_list.Size = new System.Drawing.Size(321, 68);
            this.load_list.TabIndex = 5;
            this.load_list.Text = "▶";
            this.load_list.UseVisualStyleBackColor = true;
            this.load_list.Click += new System.EventHandler(this.load_list_Click);
            // 
            // mark_fail
            // 
            this.mark_fail.AutoSize = true;
            this.mark_fail.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.mark_fail.Dock = System.Windows.Forms.DockStyle.Fill;
            this.mark_fail.Location = new System.Drawing.Point(657, 77);
            this.mark_fail.Name = "mark_fail";
            this.mark_fail.Size = new System.Drawing.Size(323, 68);
            this.mark_fail.TabIndex = 6;
            this.mark_fail.Text = "×";
            this.mark_fail.UseVisualStyleBackColor = true;
            this.mark_fail.Click += new System.EventHandler(this.mark_fail_Click);
            // 
            // tableLayoutPanel1
            // 
            this.tableLayoutPanel1.ColumnCount = 3;
            this.tableLayoutPanel1.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 33.33333F));
            this.tableLayoutPanel1.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 33.33334F));
            this.tableLayoutPanel1.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 33.33334F));
            this.tableLayoutPanel1.Controls.Add(this.prev_img, 0, 0);
            this.tableLayoutPanel1.Controls.Add(this.mark_fail, 2, 1);
            this.tableLayoutPanel1.Controls.Add(this.mark_pass, 0, 1);
            this.tableLayoutPanel1.Controls.Add(this.load_list, 1, 1);
            this.tableLayoutPanel1.Controls.Add(this.next_img, 2, 0);
            this.tableLayoutPanel1.Controls.Add(this.save_list, 1, 0);
            this.tableLayoutPanel1.Location = new System.Drawing.Point(12, 674);
            this.tableLayoutPanel1.Name = "tableLayoutPanel1";
            this.tableLayoutPanel1.RowCount = 2;
            this.tableLayoutPanel1.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 50F));
            this.tableLayoutPanel1.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 50F));
            this.tableLayoutPanel1.Size = new System.Drawing.Size(983, 148);
            this.tableLayoutPanel1.TabIndex = 7;
            // 
            // ImageBox
            // 
            this.ImageBox.Location = new System.Drawing.Point(12, 13);
            this.ImageBox.Name = "ImageBox";
            this.ImageBox.Size = new System.Drawing.Size(983, 655);
            this.ImageBox.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.ImageBox.TabIndex = 8;
            this.ImageBox.TabStop = false;
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(8F, 15F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1007, 862);
            this.Controls.Add(this.ImageBox);
            this.Controls.Add(this.tableLayoutPanel1);
            this.Name = "Form1";
            this.Text = "Form1";
            this.tableLayoutPanel1.ResumeLayout(false);
            this.tableLayoutPanel1.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.ImageBox)).EndInit();
            this.ResumeLayout(false);

        }

        #endregion
        private System.Windows.Forms.Button prev_img;
        private System.Windows.Forms.Button save_list;
        private System.Windows.Forms.Button next_img;
        private System.Windows.Forms.Button mark_pass;
        private System.Windows.Forms.Button load_list;
        private System.Windows.Forms.Button mark_fail;
        private System.Windows.Forms.TableLayoutPanel tableLayoutPanel1;
        private System.Windows.Forms.PictureBox ImageBox;
    }
}

